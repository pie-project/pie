//! `quant`'s seven JIT units, thirty-eight rows and fifteen host programs —
//! §5 step 5, and the first family whose f16 and bf16 twins are both real.
//!
//! | was | lines | is |
//! |---|---|---|
//! | `families/quant.rs` — 7 units, 38 device rows | 1,909 | this file's seven `unit!`s and two hand-joined `Unit`s |
//! | `table/quant.rs` — 11 contracts | 135 | this file's `contract!` |
//! | `driver-cuda/src/fire/dtype_cast.rs` — 2 launchers | 208 | [`cast_fp32_to_bf16`], [`scale_rows_bf16`] |
//! | `driver-cuda/src/fire/quant_int8.rs` — 3 launchers | 294 | [`quantize_bf16_to_int8_per_channel`], [`dequant_int32_w8a8_to_bf16`], [`quantize_bf16_to_fp8_e4m3_per_token_group`] |
//!
//! Nothing else changes hands: the device text stays in
//! `csrc/src/quant/*.cuh`, which is still the only definition of every
//! `__global__` in the tree.
//!
//! # THE F16/BF16 UNIT STRUCTS, WHICH THIS FAMILY IS THE FIRST TEST OF
//!
//! §5.1 recorded that rope was bf16 throughout, so §3.2's `bf16` and `f16`
//! unit structs were declared and never exercised. `quant` exercises them.
//! Eight of the thirty-eight rows are an f16 twin of a bf16 row —
//! `cast_fp32_to_f16`, `cast_f16_to_fp32`, `scale_f16`,
//! `dequant_fp8_e4m3_to_f16`, `f16_row_map_to_dense`,
//! `dequant_mxfp4_to_f16`, and the two casts whose whole subject is the
//! crossing, `cast_f16_to_bf16` and `bf16_to_fp16`.
//!
//! **They held, and they held asymmetrically.** The verdict, in the three
//! places it is decided:
//!
//!  1. **On the READ side they are distinct and the distinction is
//!     mechanical.** `*const bf16` is [`kernels::Ty::Bf16s`] and `*const f16`
//!     is `Ty::F16s`; the two are different enum variants and `Args::bind`
//!     compares them. Every row below that used to spell a read-side buffer
//!     `Buf` — eight of them — now spells the format, because the operand
//!     list is derived from the declared parameter type and there is no
//!     weaker spelling available to write by accident. That is a real
//!     tightening and it is listed under "what the derivation changed".
//!  2. **On the WRITE side `Ty` COLLAPSES THEM, and only the C++ spelling
//!     and the Rust type system do not.** There is no `Ty::Bf16sMut` and no
//!     `Ty::F16sMut`: `*mut bf16` and `*mut f16` are BOTH `Ty::BufMut`. So a
//!     port that handed a bf16 destination to `dequant_mxfp4<f16>` would
//!     marshal without complaint. What catches it is [`Abi::CPP`], which
//!     answers two different strings — `…device::bf16*` and `…device::f16*`
//!     — and the typecheck translation unit that compiles them against the
//!     instantiation. **`Abi::CPP` is load-bearing on the write side in a
//!     way it is not on the read side**, and that asymmetry is the finding.
//!  3. **In Rust, `bf16` and `f16` are distinct `#[repr(transparent)]`
//!     structs over `u16`,** so `dequant_mxfp4::<bf16>` and
//!     `::<f16>` are two functions and a caller cannot pass one for the
//!     other. This is the check that fires first and the one a reader sees.
//!
//! The measurement that says the C++ half is real is
//! [`bf16_to_fp16`](dequant_wna16)'s, transcribed from the row it replaces:
//! spelling that kernel's source operand `Buf` renders it `const f16*`
//! against a `__global__` declaring `const bf16*`, and nvcc 13.0
//! `-arch=sm_89` rejects it — *"no instance of function template … matches
//! the required type"*. `tests/device_typecheck_types.rs` compiles both.
//!
//! # `elem` IS NOT THE RUST TYPE PARAMETER, and three rows prove it
//!
//! The `unit!` binding group binds a RUST type; `elem` names the C++
//! template argument. They coincide on most rows and on three they do not,
//! which is the case a reviewer should look for first:
//!
//!  * **[`quant_fp8::raw::quant_flat`] and its siblings.**
//!    `quant_flat<Fmt>` declares `typename Fmt::store* out`, and
//!    `fp8_e4m3::store = u8` while `int8_sym::store = i8`
//!    (`quant_bf16_to_fp8.cuh:115`, `:130`). So the Rust binding is
//!    `[S = u8]` or `[S = i8]` — the STORE type, which is what crosses —
//!    and `elem` is `quant::device::fp8_e4m3` or `quant::device::int8_sym`,
//!    which is the FORMAT TAG and never crosses at all. Binding `S` to a
//!    hypothetical Rust `fp8_e4m3` would have spelled `device::fp8_e4m3*`
//!    into the typecheck TU against a C++ parameter that is `uint8_t*`.
//!  * **[`dequant_wna16::raw::bf16_to_narrow`].** The template fixes the
//!    SOURCE at bf16 and templates the destination, so `elem = device::f16`
//!    describes the far end of the cast and the near end is the concrete
//!    `*const bf16` in the parameter list. `elem` is the wrong end of this
//!    row and the row is written so that nothing derives from it.
//!  * **[`dtype_cast::raw::cast_e8m0_to`].** `elem` is
//!    `quant::device::f32` — the header's own alias, under `quant::device`
//!    and not under `device` — while the Rust binding is the primitive
//!    `f32`. Two spellings of one type, and the row states the one NVRTC
//!    resolves.
//!
//! # Seven roots, seven nested modules
//!
//! `unit!` emits `UNITS`, `ROWS`, `PARAMS` and `raw` at its invocation
//! scope, so seven invocations cannot share one. Each gets a `pub mod` and
//! [`UNITS`] is written by hand, which is the answer the macro's own doc
//! prescribes and the same shape `x::layout` uses for five. Seven is the
//! largest root count in the tree so far and the wrapper still reads: the
//! qualifier `dequant_fp8::raw::dequant_fp8_e4m3` is information a flat file
//! would have had to spell into the stub name.
//!
//! # TWO UNITS ARE HAND-JOINED, AND THAT IS THE ONE THING THE FLOOR COULD
//! NOT EXPRESS
//!
//! `quant/dequant_fp4.cuh` and `quant/dequant_wna16.cuh` each host two rows
//! whose CONTRACTS live in `table::moe` — the routed MXFP4 and W4A16 decode
//! GEMVs. Those four rows are still row-world: they carry a real
//! [`kernels::LaunchRule`], a real `Source` per operand, and `table::moe`'s
//! contract is what a trace states. `unit!` cannot express either — it emits
//! `Source::Unbound` and `LaunchRule::Unstated` by design, because a
//! fn-world row has a `fn` for its geometry — so the four are kept as
//! hand-written [`kernels::KernelSig`]s and the unit's row list is
//! CONCATENATED: the declared rows out of the nested `unit!`, then the
//! routed rows verbatim.
//!
//! **Deleting them was never an option and neither was porting them.** The
//! device row is what makes NVRTC compile the instantiation and what
//! `unit_of` answers `hosts()` from; the `table::moe` contract is what lets
//! a trace say it. A family that is half in each world needs both, and
//! `quant` is the first one. See [`dup`] for the mechanics and
//! [`ROUTED_FP4_SIGS`] / [`ROUTED_WNA16_SIGS`] for the rows.
//!
//! A `unit!` grammar for *"and these rows verbatim, with their own sigs"*
//! would remove the four-line `dup` and the two `static [DeviceKernel; 4]`s.
//! It is reported as a gap rather than taken, because the joined shape is
//! honest about what it is doing and a grammar for it would not be.
//!
//! # The renamed-symbol incident, which is why every string below is
//! verbatim
//!
//! Measured when this family first got JIT rows: `quant jit=28 aot=11
//! overlap=0`. Twenty-eight rows that compiled, resolved a lowered name on
//! an L40S, and could be fired by nothing, because every one had been given
//! a NEW name — `quant::cast_f32_to_bf16` where the table says
//! `quant::cast_fp32_to_bf16`, `quant::dequant_fp8_e4m3_bf16` where it says
//! `..._e4m3_to_bf16`, `quant::scale_bf16` standing in for
//! `quant::scale_rows_bf16`, which is not even the same kernel.
//! `examples/migration_status`, which joins the two tables ON the symbol,
//! reported `quant 0%`.
//!
//! A symbol is the string `model-compiler` writes into a trace, the key
//! `runtime::fire` looks a row up by, and the name `model-loader` calls at
//! load time. **Every symbol below is byte-identical to the one it
//! replaces**, and `driver-cuda/src/bind/mod.rs` still matches three of
//! them as literal strings.
//!
//! # What the derivation CHANGED, and it is all in one direction
//!
//! A fn-world row's operand list comes from the declared parameter types
//! through [`Abi::TY`], so eight operands that were hand-spelled `Buf`
//! became the format they actually are. `Buf` and `Bf16s` marshal
//! identically — every buffer crosses as a pointer — so nothing at fire time
//! moves; what is gained is the offline check on exactly the rows where
//! losing it means the two sixteen-bit formats are interchangeable.
//!
//! | row | was | is | why |
//! |---|---|---|---|
//! | `cast_bf16_to_fp32.src` | `Buf` | `Bf16s` | `const T*`, `T = bf16` |
//! | `cast_f16_to_fp32.src` | `Buf` | `F16s` | `const T*`, `T = f16` |
//! | `scale_bf16.src` | `Buf` | `Bf16s` | `const T*` |
//! | `scale_f16.src` | `Buf` | `F16s` | `const T*` |
//! | `scale_rows_bf16.l_bf16` | `Buf` | `Bf16s` | `const T*` |
//! | `quantize_bf16_to_mxfp4_e2m1_per_block.w_bf16` | `Buf` | `Bf16s` | `const T*` |
//! | `absmax_per_row_bf16.w` | `Buf` | `Bf16s` | `const T*` |
//! | `bf16_row_map_to_dense.raw` / `f16_row_map_to_dense.raw` | `Buf` | `Bf16s` / `F16s` | `const T*` |
//! | `dequant_wna16_int4b8_to_bf16.scale_bf16` | `Buf` | `Bf16s` | `const T*` |
//!
//! Nothing widened. There is no row where the derived type is weaker than
//! the hand-written one.
//!
//! # Geometry, and where every number came from
//!
//! Fifteen host programs, and not one grid is invented. Thirteen reproduce
//! the [`kernels::LaunchRule`] the row stated, because that is what fires
//! today and a port's first duty is to reproduce today's launches; two
//! reproduce a rectangle no rule ever stated, transcribed from the `<<<>>>`
//! it came from. Three of the fifteen fit neither [`Launch::flat`] nor
//! [`Launch::per_row`] and write the struct literal, because §5.1 says a
//! kernel that fits neither writes it.
//!
//! | host program | grid | block | smem | from |
//! |---|---|---|---|---|
//! | [`cast_fp32_to_bf16`] | `ceil(n / 256)` | 256 | 0 | `Elementwise`, `bind/launch.rs:128`; `dtype_cast.cu:51-54` |
//! | [`scale_rows_bf16`] | `rows` | `ceil_warp(width)` cap 1024 | 0 | `RouteRows`, `bind/launch.rs:157`; `dtype_cast.cu:69-72` |
//! | [`bf16_to_fp16`] | `clamp(ceil(units / 256), 1, 1024)` | 256 | 0 | `Slab`, `runtime/launch.rs:998`; `dequant_wna16.cu:63-75` |
//! | [`dequant_fp8_e4m3_to_bf16`] | `ceil(n / 256)` | 256 | 0 | `Elementwise` |
//! | [`dequant_fp8_e4m3_to_bf16_per_channel`] | `rows` | `ceil_warp(cols)` cap 1024 | 0 | `RouteRows` |
//! | [`dequant_fp8_e4m3_to_bf16_per_group`] | `rows` | `ceil_warp(cols)` cap 1024 | 0 | `RouteRows` |
//! | [`dequant_mxfp4_to_bf16`] | `rows` | `ceil_warp(in_dim)` cap 1024 | 0 | `RouteRows` |
//! | [`dequant_wna16_int4b8_to_bf16`] | `[rows, ceil(in_dim / 256)]` | 256 | 0 | `ElementwiseRows`, `bind/launch.rs:143` |
//! | [`mxfp4_scales_to_marlin_e8m0`] | `ceil(selected_rows * target_groups / 256)` | 256 | 0 | `Elementwise` |
//! | [`quantize_bf16_to_mxfp4_e2m1_per_block`] | `rows` | `ceil_warp(cols / 32)` cap 1024 | 0 | `RouteRows` |
//! | [`quantize_bf16_to_fp8_e4m3_per_channel`] | `rows` | 256 | 32 | `Rms`, `bind/launch.rs:116` |
//! | [`quantize_bf16_to_int8_per_channel`] | `rows` | 256 | 32 | `Rms`; `quant_bf16_to_fp8.cu:67-76` |
//! | [`dequant_int8_to_bf16_per_channel`] | `ceil(n / 256)` | 256 | 0 | `Elementwise` |
//! | [`dequant_int32_w8a8_to_bf16`] | `[ceil(N/32), ceil(M/8)]` | `(32, 8)` | 0 | LITERAL — `quant_bf16_to_fp8.cu:103-115` |
//! | [`quantize_bf16_to_fp8_e4m3_per_token_group`] | `[n_groups, m]` | 128 | 0 | LITERAL — `quant_bf16_to_fp8.cu:119-135` |
//!
//! The two literals are the two the row world declared
//! [`kernels::LaunchRule::Unstated`] and `fire/quant_int8.rs` stated by
//! hand: a 2-D BLOCK and a `grid.x` that is `k` divided by an OPERAND. §10.5
//! refuses vocabulary grown for one kernel, and each of those is one kernel.
//! In fn-world they need no escape hatch at all — they are two expressions
//! in two functions, which is what §5.1 means by *"the conveniences are
//! conveniences"*.
//!
//! # The three MXFP4 constants whose witness was deleted
//!
//! `quant/dequant_fp4.cu` is gone and it held three numbers. Their status,
//! established rather than re-derived:
//!
//!  * **`kMxfp4GateUpPairs = 4` and `kMxfp4DownRows = 4` are TEMPLATE
//!    ARGUMENTS**, so [`DEQUANT_FP4_ROWS`]' instantiation strings carry them
//!    (`elem: "device::i32(4)"`) and NVRTC fails loudly on a value the
//!    template rejects. **They are witnessed by the compile.** They are the
//!    same number by coincidence and not by contract — one counts gate/up
//!    PAIRS, so the warp owns `2 * kPairs` packed rows, and the other counts
//!    output ROWS — and they are spelled separately here because they are
//!    spelled separately there.
//!  * **`kMxfp4DecodeBlock = 128` is a block size the kernel ADAPTS to.**
//!    `dequant_fp4.cuh` computes its warp count as `blockDim.x >> 5` at run
//!    time, with no `__launch_bounds__` and no `static_assert`. So it is
//!    TUNING and not a correctness constraint, it has one copy and no
//!    oracle, and that is the correct shape for a tuning constant. Its
//!    provenance is `git log --follow`.
//!
//! **The argument that mattered survives where it belongs — in the kernel's
//! own comment**, `dequant_fp4.cuh:227-229`, and it is a measurement:
//!
//! > Every extra pair reuses the activation vector one more time and gives
//! > the unpack more independent work to hide behind, which is what this
//! > kernel is short of: at 2 it sustains about 1.4 TB/s against an HBM
//! > roofline near 3.
//!
//! The `4` in the two `elem` strings below is that measurement's conclusion.
//! A port that dropped it would have consumed a measurement, which is a
//! regression even when it compiles.
//!
//! # The kernels that stay without a host program, and why
//!
//! Every `__global__` in the seven roots is COMPILED — a unit compiles its
//! root — and the ones below are declared by no row and called by no `fn`.
//! Each reason is a measurement or a shape, and none is "not got to yet":
//!
//!  * **`quant_bf16_to_fp8.cuh`'s `absmax_bf16`.** A capped grid-stride
//!    whose divisor is not [`kernels::LaunchRule::Slab`]'s: `slab` divides
//!    by the eight-wide vector first because the kernel it was ported from
//!    loads `float4`s, and `absmax_bf16` strides UNVECTORISED elements at
//!    `<<<min((n + 255) / 256, 1024), 256>>>` (`quant_bf16_to_fp8.cu:40-42`).
//!    Same shape, different divisor, an eighth of the grid. It has no
//!    caller; a `fn` for it would be a first implementation, not a port.
//!  * **`dtype_cast.cuh`'s `marlin_permute_scales_per_group`.** One block
//!    per 64-scale group with a `__shared__ bf16[64]` staging buffer sized
//!    by the block. No caller.
//!  * **`dtype_cast.cuh`'s `awq_dequant_to_bf16` and `gptq_dequant_to_bf16`.**
//!    `dim3(32, 8)` blocks over `dim3(ceil(N/32), ceil(M/8))`. No caller.
//!  * **`mxfp4_marlin.cuh`'s `mxfp4_weight_to_gptq_w4`.** No caller.
//!  * **`quant_bf16_to_fp8.cu`'s fourth launcher,
//!    `launch_dequant_int8_to_bf16_per_channel`.** Deleted rather than
//!    ported, and the consumer set was swept: no `.cu`, `.cpp`, `.cuh` or
//!    `.hpp` in any archive, no table row (so `emit_c_shim` emitted no
//!    entry), no hand-written arm in `driver-cuda/src`. Its `.hpp` called it
//!    a *"correctness fallback for runtime INT8 weights when cuBLAS cannot
//!    run W8A8 for a shape"*; the fallback is `bind::quant_gemm`'s own,
//!    which is [`dequant_int8_to_bf16_per_channel`] here. The KERNEL is
//!    still declared and still fired — by that path — so this bullet is
//!    about the launcher only.
//!  * **`dequant_fp4.cuh`'s `mxfp4_moe_gate_up_decode_grouped<kTok>`.** Two
//!    independent reasons, either sufficient. Its `grid.x` is an EXPERT
//!    count where both siblings open `num_tokens * top_k`, and
//!    `Dims::n_experts` is filled with zero. And its template argument came
//!    from the ENVIRONMENT — `dequant_fp4.cu:108-135` read
//!    `std::getenv("PIE_MXFP4_MOE_KTOK")` and switched four cases with a
//!    default of 4 — so a row naming `<4>` would be right on the machines
//!    that do not set it and would silently name a different cubin entry on
//!    the machines that do. **An environment variable is not geometry.**
//!    Its own tile is `warps * kMxfp4GroupedPairs`, which at
//!    `kMxfp4DecodeBlock` is **8** against the two declared siblings' 16, so
//!    even if the expert count arrived, `RoutedQmvQuad` would state exactly
//!    twice its `grid.y` and every output row would be claimed by two
//!    blocks.
//!
//! # ELEVEN CONTRACTS, ELEVEN BINDS, AND NOT ONE `none:` ARM
//!
//! This was expected to be the family with a column of `none:`s. A
//! dequantiser's scale LOOKS like a second weight, [`Cx`] reaches weights by
//! position or by name and not by the suffix row world spelled
//! `Source::WeightSuffix`, and the driver dequantises inside
//! `bind::quant_gemm`'s GEMM rather than as a statement of its own.
//!
//! It is not. **Every one of the eleven device rows being deleted was fully
//! sourced**, and each `Source` on them has an exact [`Cx`] query behind it:
//!
//! | `Source` | `Cx` | equal because |
//! |---|---|---|
//! | `In(i)` / `Out(i)` | `arg_in(i)` / `arg_out(i)` | `Facts for Fire` indexes the same `bound.args` |
//! | `InWidth(i)` / `OutWidth(i)` | `in_width(i)` / `out_width(i)` | both are `width_of` |
//! | `Param(i)` | `param(i)` | both are `spec.params[i]` |
//! | `OutRows(0)` | `rows().count` | `bind/mod.rs:1596`'s `rows_of` DISCARDS its operand index and answers the fire's `rows`, which is the same `rows` `Fire` is built with at `bind/mod.rs:2279` |
//! | `OutElements(0)` | `rows().count * out_width(0)` | `elems_of`, verbatim |
//!
//! The scale tensors are `Source::In(1)` and not weights — the statements
//! that carry them declare two inputs — so the eleven binds below are eleven
//! transcriptions and none is a judgement call.
//!
//! Where a `none:` genuinely would have been needed is the four routed MoE
//! rows, and those are not this family's contracts: they are `table::moe`'s,
//! they stay rule-driven, and that is the whole reason this file joins two
//! row lists.
//!
//! # `Contract`: this family needed none of the ten fields
//!
//! Every one of the eleven contracts is `Contract::DEFAULT` plus a name and
//! a symbol. In particular **no `in_place`**, which is a discrepancy worth
//! stating rather than smoothing: the DEVICE rows for `scale_bf16`,
//! `scale_f16`, `scale_fp32`, `scale_rows_bf16`, `absmax_to_scale_inv_fp8`
//! and `absmax_to_scale_inv_int8` all declared `in_place = &[(0, 0)]`, and
//! `table/quant.rs` — the contract side, which is what
//! `model-compiler/src/kernels.rs:128`'s `in_place_pairs` reads for buffer
//! aliasing — declared it on none of them. A faithful port states what the
//! contract stated. Adding `in_place` here would change what the compiler
//! aliases on a path this port cannot verify, which is a different change
//! and should be made as one.
//!
//! # What breaks, stated rather than discovered
//!
//! * `tests/launch_rules.rs` pins six `quant` rows against their launchers
//!   by `LaunchRule`. Fn-world rows are `Unstated`; those assertions are
//!   about a world this family has left.
//! * `model/tests/kernels_table.rs`'s `UNSTATED_ROWS` counts rows with no
//!   rule. Thirty-eight more arrive here.
//! * `api.rs`'s generated `quant_*` entry points evaluate the row's
//!   `LaunchRule` at the fire. `model-loader` called four of them and now
//!   calls four host programs instead; any other caller of a generated
//!   `quant_*` would refuse. §2.1 lists the generated `api.rs` lines among
//!   what fn-world replaces.
//!
//! [`Abi::TY`]: crate::x::Abi::TY
//! [`Abi::CPP`]: crate::x::Abi::CPP
//! [`Cx`]: crate::x::Cx

#![allow(clippy::too_many_arguments)]

use crate::device::DeviceKernel;
use crate::unit::Unit;
use crate::x::abi::{bf16, f16};
use crate::x::launch::Launch;
use kernels::{KernelSig, LaunchRule, Lit, Source, kernel, operands};

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use core::ffi::c_void;

// ---------------------------------------------------------------------------
// Truth one, declared: seven roots, seven modules.
//
// SEVEN `unit!` INVOCATIONS CANNOT SHARE A SCOPE — each emits `UNITS`,
// `ROWS`, `PARAMS` and `raw` at its invocation scope. One `pub mod` each and
// a hand-written family `UNITS`, which is what the macro's own doc
// prescribes. `x::layout` does the same for five.
// ---------------------------------------------------------------------------

/// `quant/dtype_cast.cuh` — five cast templates and one row scaler, ten rows.
pub mod dtype_cast {
    use crate::x::abi::{bf16, f16};

    unit! {
        /// Five templates, ten instantiations, and the two ahead-of-time
        /// symbols `model-loader` calls by name.
        ///
        /// The header holds three more `__global__`s that no row declares —
        /// `marlin_permute_scales_per_group`, `awq_dequant_to_bf16` and
        /// `gptq_dequant_to_bf16` — and the unit compiles them anyway,
        /// because a unit compiles its root. See the family header for why
        /// each has no host program.
        unit DTYPE_CAST = "quant/dtype_cast",
            text = include_str!("../../csrc/src/quant/dtype_cast.cuh"),
            file = "quant/dtype_cast.cuh";

        /// `dtype_cast.cuh:104` — `dst[i] = (T)src[i]` over `n` f32.
        ///
        /// `quant::cast_fp32_to_bf16` is the loader's, called by name from
        /// Rust since before there was a JIT; the f16 twin is an
        /// instantiation the ahead-of-time build never asked for.
        fn cast_f32_to = "quant::device::cast_f32_to" <T> (
            src: *const f32,
            dst: *mut T,
            n: usize,
        ) where *mut T {
            "quant::cast_fp32_to_bf16" => where [T = bf16] "device::bf16",
            "quant::cast_fp32_to_f16" => where [T = f16] "device::f16",
        }

        /// `dtype_cast.cuh:112` — the widening direction.
        ///
        /// **`src` is where the read-side unit structs earn their keep.**
        /// The hand-written row spelled it `Buf`, which renders `const
        /// {elem}*` and is therefore right by accident; `*const T` renders
        /// the format because the format is the type.
        fn cast_to_f32 = "quant::device::cast_to_f32" <T> (
            src: *const T,
            dst: *mut f32,
            n: usize,
        ) where *const T {
            "quant::cast_bf16_to_fp32" => where [T = bf16] "device::bf16",
            "quant::cast_f16_to_fp32" => where [T = f16] "device::f16",
        }

        /// `dtype_cast.cuh:120` — f16 in, anything out.
        ///
        /// One instantiation, and the source is concrete: this template
        /// fixes the near end at f16 and templates the far end, so `elem`
        /// describes `dst`.
        fn cast_f16_to = "quant::device::cast_f16_to" <T> (
            src: *const f16,
            dst: *mut T,
            n: usize,
        ) where *mut T {
            "quant::cast_f16_to_bf16" => where [T = bf16] "device::bf16",
        }

        /// `dtype_cast.cuh:133` — an E8M0 exponent byte widened to f32.
        ///
        /// `0xFF` is NaN and every other byte is `1 << 23` shifted into the
        /// exponent field, which is the whole of the format. `elem` is
        /// `quant::device::f32` — the header's own alias, under
        /// `quant::device` rather than under `device` — while the Rust
        /// binding is the primitive `f32`.
        fn cast_e8m0_to = "quant::device::cast_e8m0_to" <T> (
            src: *const u8,
            dst: *mut T,
            n: usize,
        ) where *mut T {
            "quant::cast_e8m0_to_fp32" => where [T = f32] "quant::device::f32",
        }

        /// `dtype_cast.cuh:149` — `dst[i] = src[i] * factor`.
        ///
        /// Three instantiations and the destination type follows the source,
        /// so `T = f32` gives `Ty::F32sMut` where the two narrow formats give
        /// `Ty::BufMut`. That difference is derived, and the hand-written
        /// rows spelled exactly the same three.
        fn scale = "quant::device::scale" <T> (
            src: *const T,
            dst: *mut T,
            n: usize,
            factor: f32,
        ) where *const T, *mut T {
            "quant::scale_bf16" => where [T = bf16] "device::bf16",
            "quant::scale_f16" => where [T = f16] "device::f16",
            "quant::scale_fp32" => where [T = f32] "quant::device::f32",
        }

        /// `dtype_cast.cuh:263` — `buf[r, c] *= l[c]`, in place.
        ///
        /// `rows` is not a parameter: the `__global__` reads `blockIdx.x`
        /// and the launcher spent the number on the grid. `width` is,
        /// because `for (c = threadIdx.x; c < width; c += blockDim.x)` reads
        /// it — and that stride loop is also why the block width is the
        /// caller's to pick. `dtype_cast.cu:69-72` picked 256 and the
        /// `RouteRows` rule picks `ceil_warp(width)`; the kernel gives the
        /// same answer under both, and the paragraph saying so is the `.cu`'s
        /// own, kept here because the `.cu` is gone.
        fn scale_rows = "quant::device::scale_rows" <T> (
            buf: *mut T,
            l: *const T,
            width: i32,
        ) where *mut T, *const T {
            "quant::scale_rows_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `quant/dequant_fp8.cuh` — four scale shapes, five rows.
pub mod dequant_fp8 {
    use crate::x::abi::{bf16, f16};

    unit! {
        /// The FP8 E4M3 dequantisers: flat, per-channel, per-tile,
        /// per-group.
        ///
        /// Three of the four scale shapes are ahead-of-time symbols, because
        /// which one a checkpoint ships is a fact the declaration reads and
        /// not a fact a driver may guess — a guess dequantizes correctly on
        /// one checkpoint and silently wrongly on the next.
        unit DEQUANT_FP8 = "quant/dequant_fp8",
            text = include_str!("../../csrc/src/quant/dequant_fp8.cuh"),
            file = "quant/dequant_fp8.cuh";

        /// `dequant_fp8.cuh:88` — one f32 scale for the whole tensor.
        fn dequant_fp8_e4m3 = "quant::device::dequant_fp8_e4m3" <T> (
            src: *const u8,
            dst: *mut T,
            scale: f32,
            n: usize,
        ) where *mut T {
            "quant::dequant_fp8_e4m3_to_bf16" => where [T = bf16] "device::bf16",
            "quant::dequant_fp8_e4m3_to_f16" => where [T = f16] "device::f16",
        }

        /// `dequant_fp8.cuh:97` — one f32 scale per output channel.
        ///
        /// `rows` is `blockIdx.x` and is therefore not a parameter; the
        /// scale array is indexed by it.
        fn dequant_fp8_e4m3_per_channel = "quant::device::dequant_fp8_e4m3_per_channel" <T> (
            src: *const u8,
            dst: *mut T,
            scale_inv: *const f32,
            cols: i32,
        ) where *mut T {
            "quant::dequant_fp8_e4m3_to_bf16_per_channel" => where [T = bf16] "device::bf16",
        }

        /// `dequant_fp8.cuh:143` — a 2-D tile of scales.
        ///
        /// Declared and compiled; no contract and no host program, because
        /// nothing in the tree fires it. `row_block`, `col_block` and
        /// `scale_cols` are three parameters of the SCALE tensor's shape,
        /// which is exactly the shape no statement carries.
        fn dequant_fp8_e4m3_blocked = "quant::device::dequant_fp8_e4m3_blocked" <T> (
            src: *const u8,
            dst: *mut T,
            scales: *const f32,
            cols: i32,
            row_block: i32,
            col_block: i32,
            scale_cols: i32,
        ) where *mut T {
            "quant::dequant_fp8_e4m3_to_bf16_blocked" => where [T = bf16] "device::bf16",
        }

        /// `dequant_fp8.cuh:162` — one f32 scale per contiguous group along
        /// K, the DeepSeek block-FP8 weight layout.
        ///
        /// The kernel recomputes `scale_cols = ceil(cols / group_size)`
        /// itself, so the group size is the only extra number that crosses.
        fn dequant_fp8_e4m3_per_group = "quant::device::dequant_fp8_e4m3_per_group" <T> (
            src: *const u8,
            dst: *mut T,
            scales: *const f32,
            cols: i32,
            group_size: i32,
        ) where *mut T {
            "quant::dequant_fp8_e4m3_to_bf16_per_group" => where [T = bf16] "device::bf16",
        }
    }
}

/// `quant/quant_bf16_to_mxfp4.cuh` — the MXFP4 encoder, one row.
pub mod quant_mxfp4 {
    use crate::x::abi::bf16;

    unit! {
        /// One block per row, 32 values per E8M0 scale, two outputs.
        ///
        /// `quant_bf16_to_mxfp4.hpp:6` named its caller outright: *"Used by
        /// the Rust LoadPlan runtime quantization path: the loader emits an
        /// Encode TileMap that reads the source weight, computes absmax, and
        /// stores the quantized weight plus scale tensor directly as runtime
        /// outputs."* No row said so for a long time, so nothing could call
        /// it and the loader's Encode ran on the host against a kernel
        /// sitting unused beside it.
        unit QUANT_BF16_TO_MXFP4 = "quant/quant_bf16_to_mxfp4",
            text = include_str!("../../csrc/src/quant/quant_bf16_to_mxfp4.cuh"),
            file = "quant/quant_bf16_to_mxfp4.cuh";

        /// `quant_bf16_to_mxfp4.cuh:115` — a row to E2M1 nibbles plus its
        /// E8M0 block scales.
        ///
        /// TWO destinations — the payload and the scales it cannot be read
        /// without — which is why `TileMapOp` carries a second output. The
        /// thread's unit of work is a 32-element MXFP4 block, not an
        /// element, so a block width sized on `cols` would launch 32 threads
        /// for every one with work; the caller states `cols / 32` and the
        /// kernel's own `groups = cols / 32` performs the same truncation
        /// either way.
        fn quant_bf16_to_mxfp4_row = "quant::device::quant_bf16_to_mxfp4_row" <T> (
            src: *const T,
            packed: *mut u8,
            scales: *mut u8,
            cols: i32,
        ) where *const T {
            "quant::quantize_bf16_to_mxfp4_e2m1_per_block" => where [T = bf16] "device::bf16",
        }
    }
}

/// `quant/quant_bf16_to_fp8.cuh` — the narrow-format quantisers, eleven rows.
pub mod quant_fp8 {
    use crate::x::abi::bf16;

    unit! {
        /// Nine `__global__`s where the ahead-of-time file had twelve.
        ///
        /// It held FP8 and INT8 twins of four kernels — same body, different
        /// `max_abs` and different store type. They are one template per
        /// shape here, parameterised by a FORMAT TAG, and the row picks the
        /// format. That is where twelve became nine.
        ///
        /// **The tag never crosses.** `fp8_e4m3::store = u8` and
        /// `int8_sym::store = i8` (`:115`, `:130`), and every parameter the
        /// `__global__` declares is spelled in the store type. So the Rust
        /// binding group binds `u8` or `i8` and `elem` names the tag; see
        /// the family header, which lists this as the sharpest of the three
        /// rows where `elem` and the Rust type parameter are different
        /// things.
        unit QUANT_BF16_TO_FP8 = "quant/quant_bf16_to_fp8",
            text = include_str!("../../csrc/src/quant/quant_bf16_to_fp8.cuh"),
            file = "quant/quant_bf16_to_fp8.cuh";

        /// `quant_bf16_to_fp8.cuh:170` — `out[i] = Fmt(W[i] * scale_inv)`,
        /// one scale for the whole tensor.
        fn quant_flat = "quant::device::quant_flat" <S> (
            w: *const bf16,
            out: *mut S,
            scale_inv: f32,
            n: usize,
        ) where *mut S {
            "quant::quant_bf16_to_fp8_e4m3" => where [S = u8] "quant::device::fp8_e4m3",
        }

        /// `quant_bf16_to_fp8.cuh:185` — `x[i] = x[i] / Fmt::max_abs()`, in
        /// place.
        ///
        /// **No Rust type parameter at all**: the `__global__` takes
        /// `(float*, i32)` whichever format it is instantiated at, because
        /// the format only decides the divisor. Two rows, one parameter
        /// list, and the two differ in `elem` alone — which is the case the
        /// binding-group syntax is optional for.
        fn absmax_to_scale_inv = "quant::device::absmax_to_scale_inv" (
            x: *mut f32,
            n: i32,
        ) {
            "quant::absmax_to_scale_inv_fp8" => "quant::device::fp8_e4m3",
            "quant::absmax_to_scale_inv_int8" => "quant::device::int8_sym",
        }

        /// `quant_bf16_to_fp8.cuh:266` — INT8 back to `T`, flat, with the
        /// row recovered from the linear index.
        ///
        /// Templated on the WIDE side rather than the narrow one, and the
        /// header says why: there is no fp8 twin of this kernel to unify
        /// with, because `dequant_fp8.cuh` holds the fp8 dequantisers and
        /// they are row-shaped rather than flat.
        fn dequant_int8_per_channel = "quant::device::dequant_int8_per_channel" <T> (
            w: *const i8,
            out: *mut T,
            scale_inv: *const f32,
            cols: i32,
            n: usize,
        ) where *mut T {
            "quant::dequant_int8_to_bf16_per_channel" => where [T = bf16] "device::bf16",
        }

        /// `quant_bf16_to_fp8.cuh:198` — the per-row absmax, on its own.
        ///
        /// Stage 1 of the two-stage tensor-parallel path: the absmax is
        /// all-reduced across ranks before the scales are decided, because a
        /// rank that picked its own scale would produce a shard the others
        /// cannot be concatenated with.
        ///
        /// `smem` must hold `kBlock / 32` floats. The fold ORDER is the
        /// original's — warp shuffles down, then warp 0 over the partials —
        /// because a max over a row containing a NaN depends on it and
        /// `driver-pipeline`'s tolerance contract holds nothing about which
        /// NaN wins.
        fn absmax_per_row = "quant::device::absmax_per_row" <T> (
            w: *const T,
            absmax_out: *mut f32,
            cols: i32,
        ) where *const T {
            "quant::absmax_per_row_bf16" => where [T = bf16] "device::bf16",
        }

        /// `quant_bf16_to_fp8.cuh:234` — narrow a row AND emit its scale.
        ///
        /// The scale it writes is `absmax / Fmt::max` — the MULTIPLICATIVE
        /// factor the GEMM dispatcher hands cuBLASLt — so the dispatcher
        /// never computes a reciprocal at fire time.
        ///
        /// `extern __shared__ float warp_max[]` is sized `kBlock / 32` from
        /// the LAUNCH while the fold reads `tid < kBlock / 32` with `kBlock`
        /// a file-scope 256. A block wider than 256 reads past the array;
        /// narrower, and lanes holding partials are never folded. That is
        /// why this one is fired at `Rms`' fixed 256 and never at a width
        /// derived from the row.
        fn quant_per_channel = "quant::device::quant_per_channel" <S> (
            w: *const bf16,
            out: *mut S,
            scale_inv: *mut f32,
            cols: i32,
        ) where *mut S {
            "quant::quantize_bf16_to_fp8_e4m3_per_channel" => where [S = u8] "quant::device::fp8_e4m3",
            "quant::quantize_bf16_to_int8_per_channel" => where [S = i8] "quant::device::int8_sym",
        }

        /// `quant_bf16_to_fp8.cuh:215` — stage 2: narrow a row with a scale
        /// someone else already decided.
        ///
        /// `scale_inv` is `*const` here and `*mut` above, which is the whole
        /// difference between the two templates and is visible in the
        /// derived operand list as `F32s` against `F32sMut`.
        fn cast_per_channel = "quant::device::cast_per_channel" <S> (
            w: *const bf16,
            out: *mut S,
            scale_inv: *const f32,
            cols: i32,
        ) where *mut S {
            "quant::cast_bf16_to_fp8_e4m3_per_channel" => where [S = u8] "quant::device::fp8_e4m3",
            "quant::cast_bf16_to_int8_per_channel" => where [S = i8] "quant::device::int8_sym",
        }

        /// `quant_bf16_to_fp8.cuh:382` — the W8A8 epilogue.
        ///
        /// An `[M, N]` int32 accumulator widened to bf16 through a per-row
        /// activation scale and a per-column weight scale. One thread per
        /// output element, and deliberately not fused with the GEMM: cuBLAS
        /// writes the int32 accumulator and this scales it row by column
        /// afterwards, which is bandwidth-bound either way.
        ///
        /// Not a template — `elem` is [`DeviceKernel::PLAIN`], which says
        /// *"this `__global__` has no template parameter list"* and is the
        /// only honest thing to say about one that does not.
        ///
        /// [`DeviceKernel::PLAIN`]: crate::device::DeviceKernel::PLAIN
        fn w8a8_dequant = "quant::device::w8a8_dequant" (
            acc: *const i32,
            act_scale_inv: *const f32,
            w_scale_inv: *const f32,
            out: *mut bf16,
            m: i32,
            n: i32,
        ) {
            "quant::dequant_int32_w8a8_to_bf16" => crate::device::DeviceKernel::PLAIN,
        }

        /// `quant_bf16_to_fp8.cuh:330` — blockwise activation quantisation.
        ///
        /// One f32 scale per contiguous `gs` run along K, emitted row-major
        /// `[m, ceil(k / gs)]`, which is bit-identical to the column-major
        /// `[ceil(k / gs), m]` tensor cuBLASLt wants for
        /// `CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F` on operand B. That
        /// equality is why nothing transposes between this and the GEMM, and
        /// it is the reason the layout is not free to change.
        ///
        /// The scale is MULTIPLICATIVE — `value = fp8 * scale` — which is
        /// cuBLASLt's contract and the opposite of the per-channel weight
        /// path's `scale_inv`. Both names are the caller's; the arithmetic
        /// is what decides.
        ///
        /// `n_groups` is a parameter as well as `grid.x`: the kernel bounds
        /// `blockIdx.x` against it at `:340`. See
        /// [`quantize_bf16_to_fp8_e4m3_per_token_group`](super::quantize_bf16_to_fp8_e4m3_per_token_group)
        /// for why the quotient is computed once.
        fn quant_act_fp8_per_group = "quant::device::quant_act_fp8_per_group" (
            act: *const bf16,
            out: *mut u8,
            scale_out: *mut f32,
            m: i32,
            k: i32,
            gs: i32,
            n_groups: i32,
        ) {
            "quant::quantize_bf16_to_fp8_e4m3_per_token_group" => crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `quant/mxfp4_marlin.cuh` — the two repackers a row selector drives.
pub mod mxfp4_marlin {
    use crate::x::abi::{bf16, f16};

    unit! {
        /// Three rows over two templates.
        ///
        /// The header's third `__global__`, `mxfp4_weight_to_gptq_w4`, is
        /// compiled and undeclared: nothing in the tree fires it.
        unit MXFP4_MARLIN = "quant/mxfp4_marlin",
            text = include_str!("../../csrc/src/quant/mxfp4_marlin.cuh"),
            file = "quant/mxfp4_marlin.cuh";

        /// `mxfp4_marlin.cuh:145` — E8M0 block scales into Marlin's order.
        ///
        /// Nine integers of repack geometry after the two pointers, and they
        /// describe the SOURCE checkpoint's layout rather than the
        /// rectangle: `source_stride_groups` and `source_group_offset` are a
        /// window into a fused tensor, `row_select` is an enum the loader
        /// resolves, and `valid_rows` is how much of the selection is real.
        /// Seven of the nine are `Source::Param` — the statement carries
        /// them as wire scalars — which is why this symbol has a real bind
        /// arm and not the `none:` its shape suggests.
        ///
        /// Both buffers are `T` and `T` is `u8`, which is not a format
        /// choice: an E8M0 scale IS a byte.
        fn mxfp4_scales_to_marlin_e8m0 = "quant::device::mxfp4_scales_to_marlin_e8m0" <T> (
            raw: *const T,
            out: *mut T,
            source_rows: i32,
            source_row_offset: i32,
            selected_rows: i32,
            valid_rows: i32,
            source_stride_groups: i32,
            source_group_offset: i32,
            source_groups: i32,
            target_groups: i32,
            row_select: i32,
        ) where *const T, *mut T {
            "quant::mxfp4_scales_to_marlin_e8m0" => where [T = u8] "device::u8",
        }

        /// `mxfp4_marlin.cuh:197` — a sparse row map gathered dense.
        ///
        /// An f16/bf16 twin pair, and one of the eight rows this family
        /// contributes to the unit-struct test: the two instantiations
        /// differ in nothing but the element type, both operands are that
        /// type, and the read side now says so.
        fn row_map_to_dense = "quant::device::row_map_to_dense" <T> (
            raw: *const T,
            out: *mut T,
            batch: i32,
            source_rows: i32,
            source_row_offset: i32,
            selected_rows: i32,
            valid_rows: i32,
            row_select: i32,
        ) where *const T, *mut T {
            "quant::bf16_row_map_to_dense" => where [T = bf16] "device::bf16",
            "quant::f16_row_map_to_dense" => where [T = f16] "device::f16",
        }
    }
}

/// `quant/dequant_fp4.cuh` — the MXFP4 decoder's two rows.
///
/// **Two of this root's four rows are not here.** They are the routed MoE
/// decode GEMVs, whose contracts live in `table::moe` and whose geometry is
/// a [`LaunchRule`]; see [`DEQUANT_FP4`](super::DEQUANT_FP4), which joins
/// them onto this unit's.
pub mod dequant_fp4 {
    use crate::x::abi::{bf16, f16};

    unit! {
        /// The decoder, declared alone so the joined unit can be built from
        /// it.
        ///
        /// Named `DECODE_ONLY` and not `DEQUANT_FP4` because it is not the
        /// unit anything compiles — [`super::DEQUANT_FP4`] is, and it has
        /// four rows. This one exists to give the macro somewhere to put the
        /// two it can express.
        unit DECODE_ONLY = "quant/dequant_fp4",
            text = include_str!("../../csrc/src/quant/dequant_fp4.cuh"),
            file = "quant/dequant_fp4.cuh";

        /// `dequant_fp4.cuh:98` — packed E2M1 nibbles and E8M0 block scales
        /// to a wide row.
        ///
        /// `in_dim` is the OUTPUT width: the packed input is half as wide in
        /// bytes and the scale tensor a thirty-second, so neither input's
        /// extent is the one the kernel means. The kernel strides its block
        /// loop by `blockDim.x`, so a wider block is fewer iterations and
        /// never a different answer.
        fn dequant_mxfp4 = "quant::device::dequant_mxfp4" <T> (
            packed: *const u8,
            block_scale: *const u8,
            out: *mut T,
            in_dim: i32,
        ) where *mut T {
            "quant::dequant_mxfp4_to_bf16" => where [T = bf16] "device::bf16",
            "quant::dequant_mxfp4_to_f16" => where [T = f16] "device::f16",
        }
    }
}

/// `quant/dequant_wna16.cuh` — the W4A16 decoder and the fp16 narrowing cast.
///
/// **Two of this root's four rows are not here**, for the reason
/// [`dequant_fp4`]'s two are not: see [`DEQUANT_WNA16`](super::DEQUANT_WNA16).
pub mod dequant_wna16 {
    use crate::x::abi::{bf16, f16};

    unit! {
        /// The two rows a `fn` can fire.
        ///
        /// # THE LAUNCHER'S TWO GUARDS, WHICH NO ROW EVER MADE
        ///
        /// §54 deleted `dequant_wna16_int4b8_to_bf16`'s launcher from
        /// `kernels-cuda/csrc/src/quant/dequant_wna16.cu` — routed, no C++
        /// caller, no hand arm. It returned WITHOUT LAUNCHING on two
        /// conditions, and a rule-driven fire launched on both:
        ///
        ///   * `out_dim <= 0 || in_dim <= 0 || group_size <= 0`. The first
        ///     two are the fire's empty rectangle and are harmless;
        ///     `group_size <= 0` is not, because the kernel divides by it.
        ///   * `in_dim % 8 != 0 || in_dim % group_size != 0`. The packing is
        ///     eight 4-bit weights per `int32`, so a row whose width is not
        ///     a multiple of 8 has a partial final word the kernel reads
        ///     WHOLE — it dequantizes the padding lanes into real output
        ///     columns. A `group_size` that does not divide `in_dim` puts a
        ///     scale boundary inside a word, so the last group of each row
        ///     is scaled by its neighbour's exponent.
        ///
        /// Neither is a rectangle, so neither was statable as a
        /// [`LaunchRule`] — **and in fn-world both are, because a host
        /// program is a program.**
        /// [`dequant_wna16_int4b8_to_bf16`](super::dequant_wna16_int4b8_to_bf16)
        /// makes them, which is the first thing this port RECOVERS rather
        /// than moves. Every weight this driver has loaded satisfies both
        /// (compressed-tensors emits `group_size = 32` over an `in_dim` that
        /// is always a multiple of 128), which is why nothing has caught it.
        unit DECODE_ONLY = "quant/dequant_wna16",
            text = include_str!("../../csrc/src/quant/dequant_wna16.cuh"),
            file = "quant/dequant_wna16.cuh";

        /// `dequant_wna16.cuh:142` — INT4B8 words with a `T` scale per group
        /// along K.
        ///
        /// A different quantization from MXFP4 (an E8M0 byte per 32) and
        /// from fp8 — three quantizations, three statements, because which
        /// one a checkpoint ships is a fact the declaration reads.
        ///
        /// `packed` is `*const i32` and not a byte pointer, in the kernel
        /// and in both tables, because an INT4B8 word is eight nibbles in a
        /// 32-bit int and reading it as anything else is a stride bug a
        /// `Buf` would have allowed.
        fn dequant_wna16_int4b8 = "quant::device::dequant_wna16_int4b8" <T> (
            packed: *const i32,
            scale: *const T,
            out: *mut T,
            in_dim: i32,
            group_size: i32,
        ) where *const T, *mut T {
            "quant::dequant_wna16_int4b8_to_bf16" => where [T = bf16] "device::bf16",
        }

        /// `dequant_wna16.cuh:567` — bf16 to a narrow type, vectorised eight
        /// at a time with a scalar tail.
        ///
        /// # THIS IS THE ROW THAT NAMED `Ty::Bf16s`, AND IT IS THE FAMILY'S
        /// F16/BF16 EVIDENCE
        ///
        /// The operand was `Buf` before the units existed, exactly as the
        /// ahead-of-time twin spelled it, and `abi::device_cpp_ty` reads
        /// `Buf` as `const {elem}*` — `const f16*` for a parameter the
        /// kernel declares `const bf16*`. Every buffer marshals as a
        /// pointer, so nothing miscomputed at fire time; what was lost was
        /// the offline check, on the one row where losing it means the two
        /// sixteen-bit formats are interchangeable in the only place that
        /// distinguishes them.
        ///
        /// **`elem` is the wrong end of this cast.** `bf16_to_narrow<T>`
        /// FIXES the source at bf16 and templates the destination, so a
        /// derivation from `elem` gives `const f16*` and is wrong. Here the
        /// source is the concrete `*const bf16` in the parameter list and
        /// nothing derives it. The checker is
        /// `(const bf16*, f16*, long long)` against
        /// `bf16_to_narrow<device::f16>`, and the `Buf` spelling —
        /// `(const f16*, f16*, long long)` — is MEASURED as rejected by nvcc
        /// 13.0 `-arch=sm_89`: *"no instance of function template … matches
        /// the required type"*. `tests/device_typecheck_types.rs` compiles
        /// both.
        ///
        /// # `device::f16` and `__half` are the SAME TYPE here
        ///
        /// `csrc/shim/cuda_fp16.h` opens the NVRTC shim with `using __half =
        /// ::pie_cuda_driver::kernels::device::f16;`, so the deleted
        /// launcher's `bf16_to_narrow<__half>` and this row's
        /// `bf16_to_narrow<::pie_cuda_driver::kernels::device::f16>` name one
        /// instantiation and resolve to one `Narrow2<__half>` specialisation
        /// — the `__half2` one at `dequant_wna16.cuh:451`, which is the whole
        /// of what makes the cast fp16 rather than something else. Under
        /// nvcc the two are distinct and the header carries a
        /// `Narrow2<f16_or_inert>` behind `half_key::PickF16` so both
        /// compilers instantiate the same text at the same names; that is a
        /// parity guard and this row does not depend on it.
        ///
        /// # `n` is `i64` and it stays a parameter
        ///
        /// The grid is capped (see
        /// [`bf16_to_fp16`](super::bf16_to_fp16)), so no block can infer the
        /// extent from `gridDim`. It is `long long` and not `size_t` because
        /// the `__global__` declares `long long n` — the deleted LAUNCHER
        /// took `std::size_t count` and narrowed on the line above the
        /// `<<<>>>`, and a declaration states the kernel's parameter list
        /// rather than its caller's.
        fn bf16_to_narrow = "quant::device::bf16_to_narrow" <T> (
            in_bf16: *const bf16,
            out: *mut T,
            n: i64,
        ) where *mut T {
            "quant::bf16_to_fp16" => where [T = f16] "device::f16",
        }
    }
}

// ---------------------------------------------------------------------------
// The two hand-joined units.
//
// `unit!` states rows a `fn` can fire: every operand `Source::Unbound`, every
// geometry `LaunchRule::Unstated`, because the host program IS the binding and
// IS the geometry. Four rows in this family are not like that. They are the
// MoE decode GEMVs, their contracts live in `table::moe` — a family this port
// does not touch — and `moe`'s dispatcher fires them RULE-DRIVEN, off real
// `Source`s and a real `LaunchRule`. Dropping either would silently
// unbind a live path.
//
// So the two roots that hold them get their `unit!` rows under a nested
// `DECODE_ONLY` const and a joined `Unit` beside it, and only the joined one
// reaches `UNITS`. `unit::tests::no_symbol_is_hosted_by_two_units` is what
// makes that mandatory rather than tidy.
//
// **This is the one thing the floor could not express.** A `unit!` that took
// `+ &OTHER_ROWS` — const slice concatenation — would delete `dup`, both
// `_ROWS` statics and both joined consts. `kernelx-floor` owns the macro; this
// is the ask.
// ---------------------------------------------------------------------------

/// Copy a row out of a nested unit so a joined one can hold it.
///
/// [`DeviceKernel`] is deliberately neither `Copy` nor `Clone` — a row is a
/// singleton keyed by its symbol and duplicating one by accident is exactly
/// the defect `no_symbol_is_hosted_by_two_units` exists to catch. Here the
/// original is not in `UNITS` and the copy is, so there is still one row per
/// symbol in the set anything reads.
const fn dup(k: &DeviceKernel) -> DeviceKernel {
    DeviceKernel { sig: k.sig, template_path: k.template_path, elem: k.elem }
}

/// The two routed MXFP4 MoE decode GEMVs, whose contracts are `table::moe`'s.
///
/// The third, `mxfp4_moe_gate_up_decode_grouped`, stays without a row: its own
/// tile is `warps * kMxfp4GroupedPairs`, which is 8 against these two's 16, so
/// [`LaunchRule::RoutedQmvQuad`] is not its arithmetic — and nothing lowers it.
static ROUTED_FP4_SIGS: [KernelSig; 2] = [
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

/// `quant/dequant_fp4.cuh`'s four rows: the decoder's two, then the routed two.
static DEQUANT_FP4_ROWS: [DeviceKernel; 4] = [
    dup(&dequant_fp4::DECODE_ONLY.rows[0]),
    dup(&dequant_fp4::DECODE_ONLY.rows[1]),
    // `4` is `dequant_fp4.cu:42`'s `kMxfp4GateUpPairs` and `:44`'s
    // `kMxfp4DownRows`, and the two numbers are the SAME NUMBER by
    // coincidence rather than by contract — one counts gate/up PAIRS (so the
    // warp owns `2 * kPairs` packed rows) and the other counts output ROWS.
    // They are spelled separately here because they are spelled separately
    // there, and because a sweep that retuned one would retune it alone; the
    // shared 16-row block tile that `RoutedQmvQuad` computes is what happens
    // to fall out of both at 4.
    //
    // Both are TEMPLATE ARGUMENTS, so this string is what NVRTC instantiates
    // and a value the template rejects fails the compile loudly. That is
    // their whole oracle, and it is enough: see the family header.
    DeviceKernel {
        sig: &ROUTED_FP4_SIGS[0],
        template_path: "quant::device::mxfp4_moe_gate_up_decode",
        elem: "device::i32(4)",
    },
    DeviceKernel {
        sig: &ROUTED_FP4_SIGS[1],
        template_path: "quant::device::mxfp4_moe_down_decode",
        elem: "device::i32(4)",
    },
];

/// The MXFP4 decoder root, all four rows.
///
/// [`dequant_fp4::DECODE_ONLY`] holds the two a `fn` fires; this is the unit
/// anything compiles, resolves or typechecks against.
pub const DEQUANT_FP4: Unit = Unit {
    name: "quant/dequant_fp4",
    root: include_str!("../../csrc/src/quant/dequant_fp4.cuh"),
    rows: &DEQUANT_FP4_ROWS,
    options: &[],
};

/// The two routed W4A16 MoE decode GEMVs, whose contracts are `table::moe`'s.
static ROUTED_WNA16_SIGS: [KernelSig; 2] = [
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
    // # This row fires, and what closed it
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

/// `quant/dequant_wna16.cuh`'s four rows: the two `fn`s fire, then the routed
/// two.
static DEQUANT_WNA16_ROWS: [DeviceKernel; 4] = [
    dup(&dequant_wna16::DECODE_ONLY.rows[0]),
    dup(&dequant_wna16::DECODE_ONLY.rows[1]),
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
    // So these rows must state `0` and not the default's absence:
    // `DeviceKernel::PLAIN` would be a LIE here — it says "this `__global__`
    // has no template parameter list", and this one does — and an empty
    // `elem` is what an unfilled field looks like. The mangled name NVRTC
    // answers with carries the argument either way; the row states which one
    // it asked for.
    //
    // `device::i32(0)` rather than a bare `0` because
    // `DeviceKernel::instantiation` prefixes the FIRST token with
    // `::pie_cuda_driver::kernels::`, and `::pie_cuda_driver::kernels::0` is
    // `expected an identifier` under NVRTC 13.0 — the same measurement
    // `KV_PAGED_ROWS` records for `true`.
    DeviceKernel {
        sig: &ROUTED_WNA16_SIGS[0],
        template_path: "quant::device::wna16_gate_up_decode",
        elem: "device::i32(0)",
    },
    DeviceKernel {
        sig: &ROUTED_WNA16_SIGS[1],
        template_path: "quant::device::wna16_down_decode",
        elem: "device::i32(0)",
    },
];

/// The W4A16 decoder root, all four rows.
pub const DEQUANT_WNA16: Unit = Unit {
    name: "quant/dequant_wna16",
    root: include_str!("../../csrc/src/quant/dequant_wna16.cuh"),
    rows: &DEQUANT_WNA16_ROWS,
    options: &[],
};

/// The family's seven units, one per `.cuh`.
///
/// Hand-written because `unit!` emits a `UNITS` of its own and seven of them
/// in one module would collide; the macro's doc prescribes exactly this — a
/// nested `pub mod` per invocation and one family-level list. Five entries
/// come straight out of their module; two are the joined consts above, and
/// their nested `DECODE_ONLY`s are deliberately NOT here.
pub static UNITS: &[Unit] = &[
    dtype_cast::DTYPE_CAST,
    dequant_fp8::DEQUANT_FP8,
    quant_mxfp4::QUANT_BF16_TO_MXFP4,
    quant_fp8::QUANT_BF16_TO_FP8,
    mxfp4_marlin::MXFP4_MARLIN,
    DEQUANT_FP4,
    DEQUANT_WNA16,
];

// ---------------------------------------------------------------------------
// Geometry.
//
// Every number below is transcribed from a `<<<>>>` or from the
// `LaunchRule` the row stated, and the citation is on the constant. Nothing
// here is tuned, because nothing here was measured by this port.
// ---------------------------------------------------------------------------

/// `quant_bf16_to_fp8.cu:23` and `dtype_cast.cu:20` — `constexpr int BLOCK =
/// 256;`, twice, in two files, for two reasons.
///
/// Load-bearing for the per-channel reduction and not merely its width:
/// `quant_per_channel` sizes its `extern __shared__ float warp_max[]` at
/// `BLOCK / 32` from the LAUNCH and folds by reading `tid < kBlock / 32` with
/// `kBlock` a file-scope constant of the `.cuh`. A block wider than 256 reads
/// past the array; narrower, and lanes that hold partials are never folded.
/// That is why [`rms`] is the only geometry the two per-channel quantisers
/// are fired at.
const BLOCK: u32 = 256;

/// A warp, for the block widths that round up to one.
const WARP: u32 = 32;

/// The largest block CUDA will launch, which [`route_rows`] caps at.
const MAX_BLOCK: u32 = 1024;

/// `runtime/launch.rs:659` — [`kernels::LaunchRule::Slab`] divides by the
/// eight-wide vector before it divides by the block, because the kernel it
/// was written for loads `float4`s.
const SLAB_VEC: u32 = 8;

/// `runtime/launch.rs:668` — and then caps the grid, because a slab kernel is
/// a grid-stride loop and a bigger grid is only more blocks that exit.
const SLAB_GRID_MAX: u32 = 1024;

/// `quant_bf16_to_fp8.cu:109` — `constexpr int BX = 32, BY = 8;`, the W8A8
/// dequant's block.
///
/// A 2-D block, which is the entire reason this launch is written as a
/// literal: the kernel recovers `n` from `blockIdx.x * blockDim.x +
/// threadIdx.x` and `m` from the `y` pair, so the two axes are the two
/// extents of the output rectangle and neither can be folded into the other.
const W8A8_BX: u32 = 32;
/// The other half of the pair above.
const W8A8_BY: u32 = 8;

/// `quant_bf16_to_fp8.cu:131` — the blockwise FP8 quantiser's `128`.
///
/// One block per (row, group) and 128 threads striding the group, so this
/// width is an occupancy choice and not a contract: the kernel's loop is
/// bounded by `count = min(gs, k - base)` and strides by `blockDim.x`. It is
/// transcribed rather than rounded because the launcher chose it and nothing
/// in this port measured a different one.
const GROUP_QUANT_BLOCK: u32 = 128;

/// [`kernels::LaunchRule::Elementwise`] — `bind/launch.rs:128`.
///
/// One thread per element over a flat extent, which is
/// [`Launch::flat`] exactly.
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

/// [`kernels::LaunchRule::ElementwiseRows`] — `bind/launch.rs:143`.
///
/// A 2-D grid: one row per `grid.y` stripe of 256 columns. Neither
/// convenience states it — `flat` collapses the rectangle and `per_row`
/// gives one block a whole row — so this **writes the literal**, which is
/// what §5.1 says a kernel that fits neither does.
const fn elementwise_rows(rows: u32, width: u32) -> Launch {
    Launch { grid: [rows, width.div_ceil(BLOCK), 1], block: [BLOCK, 1, 1], smem: 0, smem_opt_in: false }
}

/// [`kernels::LaunchRule::Rms`] — `bind/launch.rs:116`.
///
/// One block per row, 256 wide, `(256 / 32) * 4` bytes of dynamic shared
/// memory: eight warp partials of four bytes each. The byte count is the
/// rule's and the kernel's `extern __shared__ float warp_max[]` is what
/// reads it.
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * 4)
}

/// [`kernels::LaunchRule::RouteRows`] — `bind/launch.rs:157`.
///
/// One block per row, as wide as the row rounded up to a warp and capped at
/// 1024. Sound only because every kernel fired at it strides its column loop
/// by `blockDim.x` rather than by the 256 the deleted launchers happened to
/// pass; the `.cu` said so beside the `<<<>>>` and the sentence is on
/// [`dtype_cast::scale_rows`]'s row.
fn route_rows(rows: u32, width: u32) -> Launch {
    Launch::per_row(rows, width.div_ceil(WARP).max(1).saturating_mul(WARP).min(MAX_BLOCK))
}

/// [`kernels::LaunchRule::Slab`] — `runtime/launch.rs:985-1015`.
///
/// A capped grid-stride loop over an eight-wide vector: `units = n / 8` when
/// there are at least eight elements and `n` when there are not, then
/// `clamp(ceil(units / 256), 1, 1024)` blocks of 256.
///
/// `quant_bf16_to_fp8.cuh`'s `absmax_bf16` is the same SHAPE with a different
/// divisor — it strides unvectorised elements — which is why it is not fired
/// through this and has no host program at all.
fn slab(n: u32) -> Launch {
    let units = if n >= SLAB_VEC { n / SLAB_VEC } else { n };
    Launch::per_row(units.div_ceil(BLOCK).clamp(1, SLAB_GRID_MAX), BLOCK)
}

/// A `usize` element count as a 32-bit launch extent, or a panic naming it.
///
/// `n` is a `usize` at the loader's boundary and a `u32` in every grid. A
/// cast that truncated would launch over the low 32 bits and leave the rest
/// of the buffer holding whatever was there — a wrong weight, silently, once
/// per load. `fire/dtype_cast.rs` made this argument and made it a panic;
/// this is the same panic with the same message, serving four host programs
/// instead of one.
///
/// # Panics
///
/// If `n` does not fit a `u32`.
#[cfg(feature = "_cuda")]
fn extent(symbol: &str, n: usize) -> u32 {
    let Ok(elems) = u32::try_from(n) else {
        panic!(
            "{symbol}: {n} elements does not fit a 32-bit launch extent; a truncating \
             cast would launch over the low 32 bits and leave the rest of the \
             destination unwritten"
        );
    };
    elems
}

// ---------------------------------------------------------------------------
// The host programs.
//
// # An empty extent DECLINES, and that is not the launchers' answer
//
// Every launcher this family replaces returned early and silently on a
// collapsed rectangle — `if (n == 0) return;`, `if (rows == 0 || cols == 0)
// return;` — and `fire/dtype_cast.rs` argued the guard back in so that a
// zero-element cast would not become a complaint: *"a zero-element cast is a
// real thing a loader asks for (an adapter with no rows for this site), and
// it was never an error."*
//
// The guards are all here, and every one of them returns
// `Fired::Declined(Refusal::Empty { .. })` rather than `Fired::Launched`,
// because `Launched` on a launch that did not happen is the one thing
// `#[must_use] enum Fired` exists to make unspellable. **The loader's
// no-op is preserved on the CALLER's side**: `Declined(Empty)` is a no-op
// there and any other refusal is an error, which is the same behaviour
// stated where the decision belongs. `rope` answers `Empty` for `num_tokens`
// in exactly this shape.
// ---------------------------------------------------------------------------

/// `dst[i] = (bf16)src[i]` for `n` fp32 elements — `quant::cast_fp32_to_bf16`.
///
/// The parameter list is `quant/dtype_cast.hpp:18-22`'s, minus nothing: the
/// two pointers, the element count and the stream, in that order, so the
/// call sites `fire/dtype_cast.rs` served did not have to move again.
///
/// Grid from [`kernels::LaunchRule::Elementwise`], which `dtype_cast.cu:51-54`
/// reproduces number for number:
///
/// ```text
/// const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
/// device::cast_f32_to<device::bf16><<<blocks, BLOCK, 0, stream>>>(
/// ```
///
/// # Panics
///
/// If `n` does not fit a `u32`. See [`extent`].
///
/// # Safety
///
/// `src_fp32` must address `n` live fp32 elements and `dst_bf16` `n` writable
/// bf16 elements, and `stream` must be live across the launch — the same
/// obligations the caller met when this was a `pie_k_*` call handing the
/// stream to a `<<<>>>`.
#[cfg(feature = "_cuda")]
pub unsafe fn cast_fp32_to_bf16(
    src_fp32: *const f32,
    dst_bf16: *mut bf16,
    n: usize,
    stream: *mut c_void,
) -> Fired {
    // `dtype_cast.cu:50`.
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    let launch = elementwise(extent("quant::cast_fp32_to_bf16", n));
    // SAFETY: the caller's obligation, above.
    unsafe {
        dtype_cast::raw::cast_f32_to::<bf16>(
            "quant::cast_fp32_to_bf16",
            launch,
            src_fp32,
            dst_bf16,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `buf[r, c] *= l[c]` over a `rows x width` bf16 buffer, in place —
/// `quant::scale_rows_bf16`.
///
/// The parameter list is `quant/dtype_cast.hpp:30-35`'s. `rows` is a
/// parameter here and NOT an argument of the launch: it is the grid, and the
/// `__global__` never took it.
///
/// Grid from [`kernels::LaunchRule::RouteRows`] against `dtype_cast.cu:69-72`'s
/// `device::scale_rows<device::bf16><<<rows, BLOCK, 0, stream>>>` — same
/// grid, and the block widths differ ON PURPOSE. The `.cu` said so beside the
/// `<<<>>>`: *"The block width is the launcher's to pick because the kernel
/// reads `blockDim.x`; 256 here, `ceil_warp(width)` under the rule, same
/// answer."* The kernel's `for (c = threadIdx.x; c < width; c +=
/// blockDim.x)` makes both exact.
///
/// # Panics
///
/// If `rows` or `width` is negative. The C++ took `int` and passed `rows`
/// straight into `<<<rows, ...>>>`, where a negative converted to an enormous
/// unsigned grid and the launch failed at the driver; saying it here names
/// the value instead.
///
/// # Safety
///
/// `buf_bf16` must address `rows * width` writable bf16 elements, `l_bf16`
/// `width` readable ones, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn scale_rows_bf16(
    buf_bf16: *mut bf16,
    l_bf16: *const bf16,
    rows: i32,
    width: i32,
    stream: *mut c_void,
) -> Fired {
    // `dtype_cast.cu:65` — `if (rows == 0 || width == 0) return;`
    if rows == 0 || width == 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    assert!(rows > 0 && width > 0, "quant::scale_rows_bf16: {rows} x {width} is not an extent");
    let launch = route_rows(rows.unsigned_abs(), width.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dtype_cast::raw::scale_rows::<bf16>(
            "quant::scale_rows_bf16",
            launch,
            buf_bf16,
            l_bf16,
            width,
            stream,
        );
    }
    Fired::Launched
}

/// Narrow a bf16 activation to fp16 — `quant::bf16_to_fp16`.
///
/// The MXFP4 MoE decode GEMVs read their activation as `__half`, so this is
/// the cast that stands between the bf16 residual stream and them. Grid from
/// [`kernels::LaunchRule::Slab`]; see [`slab`] for the arithmetic and
/// [`dequant_wna16::bf16_to_narrow`] for why `elem` describes the
/// DESTINATION and the source is concrete.
///
/// # Panics
///
/// If `n` does not fit a `u32`. See [`extent`].
///
/// # Safety
///
/// `in_bf16` must address `n` live bf16 elements, `out_fp16` `n` writable
/// fp16 elements, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn bf16_to_fp16(
    in_bf16: *const bf16,
    out_fp16: *mut f16,
    n: usize,
    stream: *mut c_void,
) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    let launch = slab(extent("quant::bf16_to_fp16", n));
    // The `__global__` takes `long long n`, and the deleted launcher took
    // `std::size_t count` and narrowed on the line above the `<<<>>>`. A
    // declaration states the kernel's parameter list, so the narrowing is
    // here.
    let count = i64::try_from(n).unwrap_or(i64::MAX);
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_wna16::raw::bf16_to_narrow::<f16>(
            "quant::bf16_to_fp16",
            launch,
            in_bf16,
            out_fp16,
            count,
            stream,
        );
    }
    Fired::Launched
}

/// One f32 scale for a whole FP8 E4M3 tensor —
/// `quant::dequant_fp8_e4m3_to_bf16`.
///
/// Grid from [`kernels::LaunchRule::Elementwise`]. `n` stays an argument as
/// well as sizing the grid because the kernel tests its own index against
/// it, which is the distinction §10.5 draws between an extent a rule
/// RECOVERS and an extent a kernel READS.
///
/// # Panics
///
/// If `n` does not fit a `u32`. See [`extent`].
///
/// # Safety
///
/// `fp8_in` addresses `n` live E4M3 bytes, `bf16_out` `n` writable bf16
/// elements, and `stream` is live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_fp8_e4m3_to_bf16(
    fp8_in: *const u8,
    bf16_out: *mut bf16,
    scale: f32,
    n: usize,
    stream: *mut c_void,
) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    let launch = elementwise(extent("quant::dequant_fp8_e4m3_to_bf16", n));
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_fp8::raw::dequant_fp8_e4m3::<bf16>(
            "quant::dequant_fp8_e4m3_to_bf16",
            launch,
            fp8_in,
            bf16_out,
            scale,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// One f32 scale per output channel —
/// `quant::dequant_fp8_e4m3_to_bf16_per_channel`.
///
/// Grid from [`kernels::LaunchRule::RouteRows`]: one block per row, the scale
/// array indexed by `blockIdx.x`, which is why `rows` is a parameter of this
/// function and not of the kernel.
///
/// # Safety
///
/// `fp8_in` addresses `rows * cols` live E4M3 bytes, `bf16_out` as many
/// writable bf16 elements, `scale_inv` `rows` f32, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_fp8_e4m3_to_bf16_per_channel(
    fp8_in: *const u8,
    bf16_out: *mut bf16,
    scale_inv: *const f32,
    rows: i32,
    cols: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 || cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_fp8::raw::dequant_fp8_e4m3_per_channel::<bf16>(
            "quant::dequant_fp8_e4m3_to_bf16_per_channel",
            launch,
            fp8_in,
            bf16_out,
            scale_inv,
            cols,
            stream,
        );
    }
    Fired::Launched
}

/// One f32 scale per contiguous group along K, the DeepSeek block-FP8 weight
/// layout — `quant::dequant_fp8_e4m3_to_bf16_per_group`.
///
/// Grid from [`kernels::LaunchRule::RouteRows`]. The kernel recomputes
/// `scale_cols = ceil(cols / group_size)` itself, so the group size is the
/// only extra number that crosses.
///
/// # Safety
///
/// `fp8_in` addresses `rows * cols` live E4M3 bytes, `bf16_out` as many
/// writable bf16 elements, `scales` `rows * ceil(cols / group_size)` f32, and
/// `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_fp8_e4m3_to_bf16_per_group(
    fp8_in: *const u8,
    bf16_out: *mut bf16,
    scales: *const f32,
    rows: i32,
    cols: i32,
    group_size: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 || cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    // The kernel divides by it.
    if group_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "group_size" });
    }
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_fp8::raw::dequant_fp8_e4m3_per_group::<bf16>(
            "quant::dequant_fp8_e4m3_to_bf16_per_group",
            launch,
            fp8_in,
            bf16_out,
            scales,
            cols,
            group_size,
            stream,
        );
    }
    Fired::Launched
}

/// Packed E2M1 nibbles and E8M0 block scales to bf16 —
/// `quant::dequant_mxfp4_to_bf16`.
///
/// Grid from [`kernels::LaunchRule::RouteRows`] where the deleted launcher
/// passed a fixed 128; the kernel strides its block loop by `blockDim.x`, so
/// a wider block is fewer iterations and never a different answer.
///
/// `in_dim` is the OUTPUT width and `out_dim` the row count: the packed input
/// is half as wide in bytes and the scale tensor a thirty-second, so neither
/// input's extent is the one the kernel means.
///
/// # Safety
///
/// `packed` addresses `out_dim * in_dim / 2` live bytes, `block_scale`
/// `out_dim * in_dim / 32`, `out` `out_dim * in_dim` writable bf16 elements,
/// and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_mxfp4_to_bf16(
    packed: *const u8,
    block_scale: *const u8,
    out: *mut bf16,
    out_dim: i32,
    in_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if out_dim <= 0 || in_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    let launch = route_rows(out_dim.unsigned_abs(), in_dim.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_fp4::raw::dequant_mxfp4::<bf16>(
            "quant::dequant_mxfp4_to_bf16",
            launch,
            packed,
            block_scale,
            out,
            in_dim,
            stream,
        );
    }
    Fired::Launched
}

/// INT4B8 words with a bf16 scale per group along K —
/// `quant::dequant_wna16_int4b8_to_bf16`.
///
/// Grid from [`kernels::LaunchRule::ElementwiseRows`] — `[rows, ceil(in_dim /
/// 256)]` blocks of 256 — which is one of the three launches here that fits
/// neither [`Launch::flat`] nor [`Launch::per_row`] and writes the literal.
///
/// # THE TWO GUARDS THIS RECOVERS
///
/// §54 deleted this kernel's launcher, which returned WITHOUT LAUNCHING on
/// two conditions a rule-driven fire cannot make, and this is where they come
/// back:
///
///   * `group_size <= 0` — the kernel DIVIDES by it.
///   * `in_dim % 8 != 0 || in_dim % group_size != 0` — the packing is eight
///     4-bit weights per `int32`, so a row whose width is not a multiple of 8
///     has a partial final word the kernel reads WHOLE and dequantizes the
///     padding lanes into real output columns; a `group_size` that does not
///     divide `in_dim` puts a scale boundary inside a word, so the last group
///     of each row is scaled by its neighbour's exponent.
///
/// Neither is a rectangle, which is why neither was statable as a
/// [`kernels::LaunchRule`], and both are two lines in a `fn`. Recovering
/// them cannot regress a caller: compressed-tensors emits `group_size = 32`
/// over an `in_dim` that is always a multiple of 128, so every weight this
/// driver has loaded satisfies both — which is also why nothing has caught
/// the missing guard.
///
/// # Safety
///
/// `packed` addresses `out_dim * in_dim / 8` live `int32`s, `scale`
/// `out_dim * in_dim / group_size` bf16, `out` `out_dim * in_dim` writable
/// bf16, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_wna16_int4b8_to_bf16(
    packed: *const i32,
    scale: *const bf16,
    out: *mut bf16,
    out_dim: i32,
    in_dim: i32,
    group_size: i32,
    stream: *mut c_void,
) -> Fired {
    if out_dim <= 0 || in_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    if group_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "group_size" });
    }
    // THE FLOOR HAS NO WORD FOR A RAGGED EXTENT, and these two guards are
    // where that shows. Neither is `Narrow` in `Refusal::Narrow`'s sense
    // ("an extent is real but below the kernel's smallest unit of work") and
    // neither is `Wide`: `in_dim` may be 4095, which is neither too small
    // nor too large but INDIVISIBLE. So each states the TAIL — the part past
    // the last whole unit — which genuinely is below one unit and makes the
    // sentence `Narrow` renders a true one. What the floor wants is a fourth
    // extent refusal, `Ragged { what, at, unit }`. `rope` and `xqa` never
    // asked for it because neither has a divisibility guard; `quant` is the
    // first family whose kernels pack.
    if in_dim % 8 != 0 {
        return Fired::Declined(Refusal::Narrow {
            what: "in_dim's tail past the last whole packed int32 word of 8 int4 values",
            at: in_dim % 8,
        });
    }
    if in_dim % group_size != 0 {
        return Fired::Declined(Refusal::Narrow {
            what: "in_dim's tail past the last whole scale group",
            at: in_dim % group_size,
        });
    }
    let launch = elementwise_rows(out_dim.unsigned_abs(), in_dim.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_wna16::raw::dequant_wna16_int4b8::<bf16>(
            "quant::dequant_wna16_int4b8_to_bf16",
            launch,
            packed,
            scale,
            out,
            in_dim,
            group_size,
            stream,
        );
    }
    Fired::Launched
}

/// E8M0 block scales into Marlin's order —
/// `quant::mxfp4_scales_to_marlin_e8m0`.
///
/// Grid from [`kernels::LaunchRule::Elementwise`] over `selected_rows *
/// target_groups`, which is the output's element count and is what the
/// deleted launcher sized its grid with.
///
/// # THE LAUNCHER'S FOUR REFUSALS, AND WHY THIS `fn` STILL DOES NOT MAKE THEM
///
/// `quant/mxfp4_marlin.cu`'s host half was not one `<<<>>>`: it `throw`ed
/// `std::runtime_error` on four conditions before launching, §54 deleted it,
/// and a rule-driven fire has made none of them since. They are recorded here
/// because a measurement a port drops is a measurement nobody can find again:
///
///   1. `validate_row_select` — `row_select` outside `{Identity, Even, Odd}`
///      threw. The kernel's `select_row` `switch` has a `default` that falls
///      through to the identity, so a bad value SILENTLY reads the wrong half
///      of an interleaved gate/up bank. The three legal values are
///      `quant/mxfp4_marlin.cuh:70-72`'s `kRowSelect*` and
///      `driver-cuda/tests/launch_abi.rs` pins them against the Rust mirror.
///   2. `source_row_offset + selected_rows * stride > source_rows` for the
///      chosen parity — a slice that runs off the end of the source bank.
///   3. `source_group_offset + target_groups > source_groups` — the same
///      check on the group axis, which is where a tensor-parallel shard of
///      the scale table is taken.
///   4. `total % 64 != 0` — Marlin's E8M0 scale tile is 64 bytes wide and the
///      kernel writes whole tiles, so a total that is not a multiple of 64
///      leaves a partial tile of uninitialised scales that the GEMM then
///      reads as exponents.
///
/// A `fn` CAN make all four, where a [`kernels::LaunchRule`] could make none
/// — a rule states a rectangle, not a predicate — so this is now the place
/// they belong. They are still not made, and the reason is the same one that
/// keeps `in_place` off this family's contracts: three of the four would turn
/// a fire that works today into a refusal on a path this port cannot verify.
/// [`dequant_wna16_int4b8_to_bf16`]'s two guards ARE recovered, and the
/// difference is that no caller can be relying on their absence — a zero
/// divisor and a partial word are wrong for every input. Adding these four is
/// a separate change and should be made as one, with a caller that knows the
/// shard.
///
/// # Safety
///
/// `raw` addresses `source_rows * source_stride_groups` live E8M0 bytes, `out`
/// `selected_rows * target_groups` writable ones, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn mxfp4_scales_to_marlin_e8m0(
    raw: *const u8,
    out: *mut u8,
    source_rows: i32,
    source_row_offset: i32,
    selected_rows: i32,
    valid_rows: i32,
    source_stride_groups: i32,
    source_group_offset: i32,
    source_groups: i32,
    target_groups: i32,
    row_select: i32,
    stream: *mut c_void,
) -> Fired {
    if selected_rows <= 0 || target_groups <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the repacked rectangle" });
    }
    let total = selected_rows.unsigned_abs().saturating_mul(target_groups.unsigned_abs());
    let launch = elementwise(total);
    // SAFETY: the caller's obligation, above.
    unsafe {
        mxfp4_marlin::raw::mxfp4_scales_to_marlin_e8m0::<u8>(
            "quant::mxfp4_scales_to_marlin_e8m0",
            launch,
            raw,
            out,
            source_rows,
            source_row_offset,
            selected_rows,
            valid_rows,
            source_stride_groups,
            source_group_offset,
            source_groups,
            target_groups,
            row_select,
            stream,
        );
    }
    Fired::Launched
}

/// A bf16 rectangle to MXFP4 nibbles plus their E8M0 block scales —
/// `quant::quantize_bf16_to_mxfp4_e2m1_per_block`.
///
/// Grid from [`kernels::LaunchRule::RouteRows`], and the width the rule is
/// given is `cols / 32` and not `cols`: the thread's unit of work is a
/// 32-element MXFP4 block, so a block width sized on `cols` would launch 32
/// threads for every one with work. The kernel's own `groups = cols / 32`
/// performs the same truncation either way.
///
/// TWO destinations — the payload and the scales it cannot be read without —
/// which is why `TileMapOp` carries a second output.
///
/// # Safety
///
/// `w_bf16` addresses `rows * cols` live bf16, `w_packed` `rows * cols / 2`
/// writable bytes, `w_scale_e8m0` `rows * cols / 32` writable bytes, and
/// `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn quantize_bf16_to_mxfp4_e2m1_per_block(
    w_bf16: *const bf16,
    w_packed: *mut u8,
    w_scale_e8m0: *mut u8,
    rows: i32,
    cols: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 || cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    if cols < 32 {
        return Fired::Declined(Refusal::Narrow { what: "cols, in 32-element blocks", at: cols });
    }
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs() / 32);
    // SAFETY: the caller's obligation, above.
    unsafe {
        quant_mxfp4::raw::quant_bf16_to_mxfp4_row::<bf16>(
            "quant::quantize_bf16_to_mxfp4_e2m1_per_block",
            launch,
            w_bf16,
            w_packed,
            w_scale_e8m0,
            cols,
            stream,
        );
    }
    Fired::Launched
}

/// Per-row FP8 E4M3 quantisation with the scale emitted beside it —
/// `quant::quantize_bf16_to_fp8_e4m3_per_channel`.
///
/// Grid from [`kernels::LaunchRule::Rms`]: one block per row, 256 wide,
/// `(256 / 32) * 4` bytes of shared memory. The 256 is not a tuning choice
/// here — see [`BLOCK`], and the kernel's `extern __shared__ float
/// warp_max[]` is what reads the byte count.
///
/// The scale it writes is `absmax / max_abs`, the MULTIPLICATIVE factor the
/// GEMM dispatcher hands cuBLASLt, so the dispatcher never computes a
/// reciprocal at fire time.
///
/// # Safety
///
/// `w_bf16` addresses `rows * cols` live bf16, `w_fp8` as many writable
/// bytes, `scale_inv` `rows` writable f32, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn quantize_bf16_to_fp8_e4m3_per_channel(
    w_bf16: *const bf16,
    w_fp8: *mut u8,
    scale_inv: *mut f32,
    rows: i32,
    cols: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 || cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    let launch = rms(rows.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        quant_fp8::raw::quant_per_channel::<u8>(
            "quant::quantize_bf16_to_fp8_e4m3_per_channel",
            launch,
            w_bf16,
            w_fp8,
            scale_inv,
            cols,
            stream,
        );
    }
    Fired::Launched
}

/// Per-row symmetric INT8 quantisation of a `[rows, cols]` bf16 rectangle —
/// `quant::quantize_bf16_to_int8_per_channel`.
///
/// One scale per row, `scale_inv_row = absmax / 127`, outliers clamped — the
/// same op whether the rows are output channels of a weight or tokens of an
/// activation, which is why `quantize_bf16_to_int8_per_token` was a C++
/// forwarder onto this one and why there is one function here.
///
/// `quant_bf16_to_fp8.cu:67-76`, which is [`kernels::LaunchRule::Rms`] number
/// for number:
///
/// ```text
/// if (rows == 0 || cols == 0) return;
/// device::quant_per_channel<device::int8_sym>
///     <<<rows, BLOCK, ROW_REDUCE_SHMEM, stream>>>(
///         static_cast<const device::bf16*>(W_bf16), W_int8, scale_inv_dev, cols);
/// ```
///
/// The `<=` in the guard below is transcribed from `== 0` for the reason
/// every ported guard is: the C++ took `int` and a negative would have
/// produced an enormous unsigned grid, which the original only avoided
/// because no caller passed one.
///
/// **`S = i8` and not `u8`**, and `elem` is `quant::device::int8_sym` and not
/// a Rust type name: the `__global__` declares `typename Fmt::store*` and
/// `int8_sym::store = i8` at `quant_bf16_to_fp8.cuh:130`. The FP8 twin above
/// is the same template at `u8`. That pair is the sharpest case in this
/// family of `elem` and the Rust type parameter being different things.
///
/// # Safety
///
/// `w_bf16` addresses `rows * cols` live bf16, `out_int8` as many writable
/// **signed** bytes, `scale_inv` `rows` writable f32, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn quantize_bf16_to_int8_per_channel(
    w_bf16: *const bf16,
    out_int8: *mut i8,
    scale_inv: *mut f32,
    rows: i32,
    cols: i32,
    stream: *mut c_void,
) -> Fired {
    // `quant_bf16_to_fp8.cu:71`.
    if rows <= 0 || cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    let launch = rms(rows.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        quant_fp8::raw::quant_per_channel::<i8>(
            "quant::quantize_bf16_to_int8_per_channel",
            launch,
            w_bf16,
            out_int8,
            scale_inv,
            cols,
            stream,
        );
    }
    Fired::Launched
}

/// INT8 back to bf16 through a per-channel scale —
/// `quant::dequant_int8_to_bf16_per_channel`.
///
/// Grid from [`kernels::LaunchRule::Elementwise`]: flat over `rows * cols`,
/// with the row recovered from the linear index, which is why `cols` crosses
/// as well as `n`.
///
/// This is `bind::quant_gemm`'s correctness fallback for runtime INT8 weights
/// when cuBLAS cannot run W8A8 for a shape. The `.cu` had a fourth launcher
/// for it, `launch_dequant_int8_to_bf16_per_channel`, and that launcher had
/// an empty consumer set and was deleted rather than ported — the KERNEL is
/// this, and it is fired.
///
/// # Panics
///
/// If `rows * cols` does not fit a `u32`. See [`extent`].
///
/// # Safety
///
/// `w_int8` addresses `rows * cols` live signed bytes, `out` as many writable
/// bf16, `scale_inv` `rows` f32, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_int8_to_bf16_per_channel(
    w_int8: *const i8,
    out: *mut bf16,
    scale_inv: *const f32,
    rows: i32,
    cols: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 || cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    let n = rows.unsigned_abs() as usize * cols.unsigned_abs() as usize;
    let launch = elementwise(extent("quant::dequant_int8_to_bf16_per_channel", n));
    // SAFETY: the caller's obligation, above.
    unsafe {
        quant_fp8::raw::dequant_int8_per_channel::<bf16>(
            "quant::dequant_int8_to_bf16_per_channel",
            launch,
            w_int8,
            out,
            scale_inv,
            cols,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// The W8A8 epilogue: an `[M, N]` int32 accumulator widened to bf16 through a
/// per-row activation scale and a per-column weight scale —
/// `quant::dequant_int32_w8a8_to_bf16`.
///
/// **THE LITERAL**, and the first of the two. `quant_bf16_to_fp8.cu:103-115`,
/// transcribed digit for digit:
///
/// ```text
/// if (M == 0 || N == 0) return;
/// constexpr int BX = 32, BY = 8;
/// const dim3 block(BX, BY);
/// const dim3 grid((N + BX - 1) / BX, (M + BY - 1) / BY);
/// device::w8a8_dequant<<<grid, block, 0, stream>>>(
///     acc_int32, act_scale_inv, w_scale_inv,
///     static_cast<device::bf16*>(out_bf16), M, N);
/// ```
///
/// A 2-D BLOCK, which no [`Launch`] convenience states and no
/// [`kernels::LaunchRule`] ever did — the row world declared it
/// [`kernels::LaunchRule::Unstated`] and `fire/quant_int8.rs` reached for
/// `fire::hand` to state it. Here it is an expression in a function, which is
/// what §5.1 means by *"the conveniences are conveniences"*.
///
/// `M` and `N` cross as arguments as well as sizing the grid because the
/// kernel's `if (n >= N || m >= M) return;` is what stops the last block of
/// each axis, exactly as the launcher left it.
///
/// # Safety
///
/// `acc` addresses `m * n` live i32, `act_scale_inv` `m` f32, `w_scale_inv`
/// `n` f32, `out` `m * n` writable bf16, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_int32_w8a8_to_bf16(
    acc: *const i32,
    act_scale_inv: *const f32,
    w_scale_inv: *const f32,
    out: *mut bf16,
    m: i32,
    n: i32,
    stream: *mut c_void,
) -> Fired {
    // `:108` — `if (M == 0 || N == 0) return;`
    if m <= 0 || n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the accumulator" });
    }
    let launch = Launch {
        grid: [n.unsigned_abs().div_ceil(W8A8_BX), m.unsigned_abs().div_ceil(W8A8_BY), 1],
        block: [W8A8_BX, W8A8_BY, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        quant_fp8::raw::w8a8_dequant(
            "quant::dequant_int32_w8a8_to_bf16",
            launch,
            acc,
            act_scale_inv,
            w_scale_inv,
            out,
            m,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// Blockwise (per-token-group) FP8 E4M3 activation quantisation — the
/// activation half of DeepSeek-style block FP8,
/// `quant::quantize_bf16_to_fp8_e4m3_per_token_group`.
///
/// **THE LITERAL**, and the second of the two. `quant_bf16_to_fp8.cu:119-135`:
///
/// ```text
/// if (m <= 0 || k <= 0 || group_size <= 0) return;
/// const int n_groups = (k + group_size - 1) / group_size;
/// const dim3 grid(static_cast<unsigned>(n_groups), static_cast<unsigned>(m));
/// device::quant_act_fp8_per_group<<<grid, 128, 0, stream>>>(
///     static_cast<const device::bf16*>(act_bf16),
///     act_fp8, act_scale, m, k, group_size, n_groups);
/// CUDA_CHECK(cudaGetLastError());
/// ```
///
/// `grid.x` is `k` divided by an OPERAND, which is the reason this one was
/// [`kernels::LaunchRule::Unstated`] too: §10.5 refuses vocabulary grown for
/// one kernel, and this is one kernel.
///
/// `n_groups` is computed ONCE and used twice — it is `grid.x` and it is the
/// argument the kernel bounds `blockIdx.x` against at
/// `quant_bf16_to_fp8.cuh:340`. Two derivations of one quotient is how a grid
/// and a guard come to disagree, so the port keeps the single binding. The
/// quotient is taken in `i32` as the C++ had it, so a `k` near `i32::MAX`
/// overflows here exactly where it overflowed there rather than silently
/// differing.
///
/// The scale is MULTIPLICATIVE — `value = fp8 * scale` — which is cuBLASLt's
/// contract and the opposite of the per-channel weight path's `scale_inv`.
/// Both names are the caller's; the arithmetic is what decides.
///
/// The launcher's trailing `CUDA_CHECK(cudaGetLastError())` has no
/// transcription and does not need one: [`crate::x::fire::fire`] panics with
/// the symbol named on any launch error, which is the same claim with more
/// information in it.
///
/// # Safety
///
/// `act_bf16` addresses `m * k` live bf16, `act_fp8` as many writable bytes,
/// `act_scale` `m * ceil(k / group_size)` writable f32, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn quantize_bf16_to_fp8_e4m3_per_token_group(
    act_bf16: *const bf16,
    act_fp8: *mut u8,
    act_scale: *mut f32,
    m: i32,
    k: i32,
    group_size: i32,
    stream: *mut c_void,
) -> Fired {
    // `:128` — and the third term guards the DIVISION below, not a grid.
    if m <= 0 || k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the activation" });
    }
    if group_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "group_size" });
    }
    // `:129`.
    let n_groups = (k + group_size - 1) / group_size;
    let launch = Launch {
        grid: [n_groups.unsigned_abs(), m.unsigned_abs(), 1],
        block: [GROUP_QUANT_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        quant_fp8::raw::quant_act_fp8_per_group(
            "quant::quantize_bf16_to_fp8_e4m3_per_token_group",
            launch,
            act_bf16,
            act_fp8,
            act_scale,
            m,
            k,
            group_size,
            n_groups,
            stream,
        );
    }
    Fired::Launched
}

// ---------------------------------------------------------------------------
// The contracts.
//
// Eleven, which is `table/quant.rs`'s eleven, symbol for symbol and `name`
// for `name`. `Contract::DEFAULT` supplies the other ten fields; see the
// family header for why `in_place` is not among them.
// ---------------------------------------------------------------------------

contract! {
    /// 4-bit weights with a bf16 scale per group along K.
    DEQUANT_WNA16_INT4B8_TO_BF16 = "quant::dequant_wna16_int4b8_to_bf16" as dequant_wna16_int4b8

    /// The loader's narrowing cast, called by name from Rust since before
    /// there was a JIT.
    CAST_FP32_TO_BF16 = "quant::cast_fp32_to_bf16" as cast_f32_to_bf16

    /// E8M0 block scales repacked into Marlin's order.
    MXFP4_SCALES_TO_MARLIN_E8M0 = "quant::mxfp4_scales_to_marlin_e8m0" as mxfp4_scales_to_marlin

    /// One f32 scale for the whole tensor.
    DEQUANT_FP8_E4M3_TO_BF16 = "quant::dequant_fp8_e4m3_to_bf16" as dequant_fp8_e4m3

    /// One f32 scale per output channel.
    DEQUANT_FP8_E4M3_TO_BF16_PER_CHANNEL = "quant::dequant_fp8_e4m3_to_bf16_per_channel"
        as dequant_fp8_e4m3_per_channel

    /// One f32 scale per contiguous group along K.
    DEQUANT_FP8_E4M3_TO_BF16_PER_GROUP = "quant::dequant_fp8_e4m3_to_bf16_per_group"
        as dequant_fp8_e4m3_per_group

    /// MXFP4 nibbles and their E8M0 block scales, widened.
    DEQUANT_MXFP4_TO_BF16 = "quant::dequant_mxfp4_to_bf16" as dequant_mxfp4

    /// The activation cast the MXFP4 MoE decode GEMVs read through.
    BF16_TO_FP16 = "quant::bf16_to_fp16" as bf16_to_fp16

    /// Fold a per-column vector into a weight after a merge.
    SCALE_ROWS_BF16 = "quant::scale_rows_bf16" as scale_rows

    /// The loader's Encode path, MXFP4 half — two outputs.
    QUANTIZE_BF16_TO_MXFP4_E2M1_PER_BLOCK = "quant::quantize_bf16_to_mxfp4_e2m1_per_block"
        as quantize_bf16_to_mxfp4

    /// The loader's Encode path, FP8 half — two outputs.
    QUANTIZE_BF16_TO_FP8_E4M3_PER_CHANNEL = "quant::quantize_bf16_to_fp8_e4m3_per_channel"
        as quantize_bf16_to_fp8_per_channel
}

// ---------------------------------------------------------------------------
// What a trace's statement fires.
//
// Eleven arms, no `none:`. Each body is the deleted device row's `Source`
// list read through `Cx`, in the row's order, and nothing else — see the
// family header's table for why each pair is an equality rather than a
// resemblance.
//
// `//` and not `///`: these are array elements and Rust has no attributes
// there.
// ---------------------------------------------------------------------------

#[cfg(feature = "_cuda")]
bind! {
    DEQUANT_WNA16_INT4B8_TO_BF16 => { cx, stream => {
        // `group_size: I32 <- Source::Param(0)`, converted the way
        // `abi.rs:1119` converts every `Source::Param`:
        // `i32::try_from(..).unwrap_or(0)`. A 0 lands on the host program's
        // own `group_size <= 0` guard, which is the kernel's divisor.
        let group_size = i32::try_from(cx.param(0)?).unwrap_or(0);
        unsafe {
            dequant_wna16_int4b8_to_bf16(
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                group_size,
                stream,
            )
        }
        .ok()
    }},

    CAST_FP32_TO_BF16 => { cx, stream => {
        // `n: Usize <- Source::OutElements(0)` is `elems_of(b, n_in, rows)`,
        // which is `rows * width_of(out 0)` — the product, taken here.
        let rows = cx.rows().count;
        let width = cx.out_width(0)?;
        if rows <= 0 || width <= 0 {
            return Err(Refusal::Empty { what: "the output rectangle" });
        }
        let n = rows.unsigned_abs() as usize * width.unsigned_abs() as usize;
        unsafe {
            cast_fp32_to_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<bf16>(),
                n,
                stream,
            )
        }
        .ok()
    }},

    MXFP4_SCALES_TO_MARLIN_E8M0 => { cx, stream => {
        // Seven `Source::Param`s and two extents. `row_select` is
        // `Mxfp4RowSelect` in the deleted launcher and an `int` here: the
        // enum is declared in `mxfp4_marlin.hpp`, which NVRTC never sees,
        // and `enum class ... : int` makes the cast exact.
        let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
        unsafe {
            mxfp4_scales_to_marlin_e8m0(
                cx.arg_in(0)?.cast_const().cast::<u8>(),
                cx.arg_out(0)?.cast::<u8>(),
                param(0)?,
                param(1)?,
                cx.rows().count,
                param(2)?,
                param(3)?,
                param(4)?,
                param(5)?,
                cx.out_width(0)?,
                param(6)?,
                stream,
            )
        }
        .ok()
    }},

    DEQUANT_FP8_E4M3_TO_BF16 => { cx, stream => {
        // THE ROW SAID `scale: F32 <- Source::Param(0)`, AND THAT PAIR
        // CANNOT BE GENERATED. `abi.rs:966` spells an `F32` operand
        // `ArgValue::F32(..)` and `abi.rs:1119` spells a `Source::Param`
        // `i32::try_from(spec.params[0]).unwrap_or(0)`, and `cast_for` adds
        // no conversion because `Ty::F32` is not one of its five integral
        // kinds — so the composition is an `i32` handed to a variant that
        // takes an `f32`. The live caller is `bind::quant_gemm`, which fires
        // this symbol by hand with an explicit `ArgValue::F32`, so the
        // mis-stated pair never had to compose.
        //
        // A float statement parameter is its bit pattern, which is
        // `Source::ParamF32` and is `Cx::param_f32`. That is what this
        // reads. The alternative — reproducing the row — is reproducing a
        // scale of `1` for every checkpoint whose scale is not an integer,
        // which is all of them.
        let rows = cx.rows().count;
        let width = cx.out_width(0)?;
        if rows <= 0 || width <= 0 {
            return Err(Refusal::Empty { what: "the output rectangle" });
        }
        let n = rows.unsigned_abs() as usize * width.unsigned_abs() as usize;
        unsafe {
            dequant_fp8_e4m3_to_bf16(
                cx.arg_in(0)?.cast_const().cast::<u8>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.param_f32(0)?,
                n,
                stream,
            )
        }
        .ok()
    }},

    DEQUANT_FP8_E4M3_TO_BF16_PER_CHANNEL => { cx, stream => {
        // `rows` is the rule's grid and not a kernel parameter; the host
        // program takes it because a host program IS the grid.
        unsafe {
            dequant_fp8_e4m3_to_bf16_per_channel(
                cx.arg_in(0)?.cast_const().cast::<u8>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    DEQUANT_FP8_E4M3_TO_BF16_PER_GROUP => { cx, stream => {
        let group_size = i32::try_from(cx.param(0)?).unwrap_or(0);
        unsafe {
            dequant_fp8_e4m3_to_bf16_per_group(
                cx.arg_in(0)?.cast_const().cast::<u8>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.rows().count,
                cx.out_width(0)?,
                group_size,
                stream,
            )
        }
        .ok()
    }},

    DEQUANT_MXFP4_TO_BF16 => { cx, stream => {
        // `in_dim` is `OutWidth(0)` and not either input's width: the packed
        // input is half as wide in bytes and the scale tensor a
        // thirty-second.
        unsafe {
            dequant_mxfp4_to_bf16(
                cx.arg_in(0)?.cast_const().cast::<u8>(),
                cx.arg_in(1)?.cast_const().cast::<u8>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    BF16_TO_FP16 => { cx, stream => {
        let rows = cx.rows().count;
        let width = cx.out_width(0)?;
        if rows <= 0 || width <= 0 {
            return Err(Refusal::Empty { what: "the output rectangle" });
        }
        let n = rows.unsigned_abs() as usize * width.unsigned_abs() as usize;
        unsafe {
            bf16_to_fp16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<f16>(),
                n,
                stream,
            )
        }
        .ok()
    }},

    SCALE_ROWS_BF16 => { cx, stream => {
        // `buf_bf16 <- Source::Out(0)` and the row declared `in_place =
        // &[(0, 0)]`, so the buffer this scales is the OUTPUT slot and the
        // input aliases it. `l_bf16 <- Source::In(1)` is the second input,
        // which is the per-column vector.
        unsafe {
            scale_rows_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    QUANTIZE_BF16_TO_MXFP4_E2M1_PER_BLOCK => { cx, stream => {
        // TWO OUTPUTS, and `cols` is `InWidth(0)` rather than an output's:
        // the packed output is half as wide in bytes and the scale output a
        // thirty-second, so the source row is the only extent that means
        // elements.
        unsafe {
            quantize_bf16_to_mxfp4_e2m1_per_block(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<u8>(),
                cx.arg_out(1)?.cast::<u8>(),
                cx.rows().count,
                cx.in_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    QUANTIZE_BF16_TO_FP8_E4M3_PER_CHANNEL => { cx, stream => {
        // Two outputs again: the narrow row and the MULTIPLICATIVE
        // `weight_scale_inv` the GEMM dispatcher hands cuBLASLt.
        unsafe {
            quantize_bf16_to_fp8_e4m3_per_channel(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<u8>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                cx.in_width(0)?,
                stream,
            )
        }
        .ok()
    }},
}
