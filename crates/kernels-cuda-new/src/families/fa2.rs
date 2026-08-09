//! The FA2 lattice: FlashInfer's two paged `__global__`s, as JIT units.
//!
//! # What this file replaces
//!
//! `kernels-cuda/csrc/src/kernels.def` and the four thirteen-line translation
//! units that `#include` it:
//!
//! ```text
//! attention_flashinfer_hd64.cu    template struct AttnHd<64>;
//! attention_flashinfer_hd128.cu   template struct AttnHd<128>;
//! attention_flashinfer_hd256.cu   template struct AttnHd<256>;
//! attention_flashinfer_hd512.cu   template struct AttnHd<512>;
//! ```
//!
//! **All four are now deleted.** This block is their record, and the reason
//! they went without any of the three shim-entry mechanisms is that they never
//! had a row to lose: `AttnHd<HD>` is a C++ class template, not a table
//! symbol, so `abi::emit_c_shim` never wrote a `pie_k_*` for one and nothing
//! in `crates/model/src` names either the type or a wrapper for it. They were
//! reached only by nvcc being told to compile them.
//!
//! `kernels.def` itself STAYS. It is not a translation unit and it has two
//! consumers that are not compilers: `csrc/src/kernels_manifest.hpp`'s
//! `head_dim_instantiated()`, which is what keeps "96 is deliberately absent"
//! a checkable claim rather than a remark, and [`HEAD_DIMS`] below. What was
//! retired with the files is the two-way consistency check in
//! `csrc/CMakeLists.txt` that made `kernels.def`'s list and the set of
//! `attention_flashinfer_hd*.cu` on disk agree. That check could not simply be
//! left: `kernels.def` still declares all four head dims and none of the four
//! files exists, so its forward loop would stop configure with a `FATAL_ERROR`
//! whose remedy -- "copy an existing stub" -- is now precisely the wrong
//! advice. It was retired with its reasoning, not deleted.
//!
//! Each one existed to make nvcc run FlashInfer's host derivations ahead of
//! time and emit every instantiation the run-time arms might select. That is
//! what a lattice declaration is, and it is what this file is: the same
//! products, stated as data, compiled by NVRTC on the first fire that reaches
//! one.
//!
//! # The axes, and where each came from in `kernels.def`
//!
//! `kernels.def` is a two-macro file and both macros are here:
//!
//! | `kernels.def` | here | points |
//! |---|---|---|
//! | `PIE_ATTN_HEAD_DIM(HD)` | [`HEAD_DIMS`] | 64, 128, 256, 512 |
//! | `PIE_ATTN_DECODE_GQA(G)` | [`DECODE_GQA`] | 1, 2, 3, 4, 8 |
//!
//! and it names, for each point, the checkpoints it serves. Those comments are
//! carried onto the units below verbatim in substance, because they are the
//! only record of WHY a point is in the list:
//!
//! * **64** — llama3.2-1B.
//! * **128** — qwen3, qwen2, llama, mistral, olmo3; phi3, whose 96 is padded up
//!   to 128 by `head_dim_pad` rather than instantiated.
//! * **256** — gemma2, gemma3, gemma3n, gemma4's sliding layers.
//! * **512** — gemma4's global layers, deepseek v2/v3/v4's MLA absorb path,
//!   kimi_k2.
//!
//! `kernels.def` argues that **96 is deliberately absent**: phi3 is padded, and
//! a head dim that no checkpoint reaches unpadded is an instantiation nobody
//! fires. That argument is unchanged here and is now cheaper to hold, because
//! an absent point costs nothing at all rather than costing a `#include`.
//!
//! GQA groups **5, 6 and 7 are absent by the same argument in the other
//! direction**: `force_prefill_path` routes them to the prefill kernel, so a
//! decode instantiation for them would be dead. Under the archive their absence
//! was a LINK error waiting to happen if the router ever changed; here it is
//! [`crate::fa2::Refusal::DecodeGroupSize`], raised at the fire with the group
//! in the message. That is the first of the four arguments whose MEANING
//! changes, and the change is an improvement.
//!
//! # The four `kernels.def` arguments, and what a JIT does to each
//!
//! **1. "The head_dim list is permissive."** `kernels.def` instantiates head
//! dims no shipped checkpoint uses, on the argument that a missing one is a
//! link error at model load and the cost is only build time. Under a JIT the
//! cost side vanishes — [`UNITS`] holds 64, 128, 256 and 512 and a run that
//! serves llama3.2-1B compiles the four units its head dim needs and none of
//! the other 52. **The argument survives and its price is now zero**, so the
//! list should stay permissive and should grow rather than shrink.
//!
//! **2. "Runtime-policy axes stay fully instantiated."** `CTA_TILE_Q` comes
//! from `plan_info` and varies with the batch; the mask mode and the variant
//! vary with the request. `kernels.def` therefore refuses to prune them. That
//! argument is UNCHANGED and is the reason a prefill unit here holds all ten
//! (mask, variant, params) triples rather than one: they are chosen inside a
//! single fire, so a unit that held one would be a compile per request.
//!
//! **3. "The TU split took the build from 318 s to 79 s."** This is the one
//! whose beneficiary changes completely. The 318→79 measurement was about
//! `make -j`: four translation units compile on four cores where one compiles
//! on one, and the wall clock is the longest TU rather than the sum. **A JIT
//! has no `-j` and no wall clock to shorten** — it compiles one unit, on
//! demand, in the process that fires it, and the number that matters is the
//! cost of THAT ONE compile on the first fire. So the split survives but its
//! justification is replaced: units are split here so that **a fire compiles
//! only the instantiations it can select**, and the granularity is chosen to
//! keep one compile near one FlashInfer kernel's cost (~187 KB of PTX for
//! decode, ~146 KB for prefill, both measured by `nvrtcGetPTXSize` on this
//! box). Four units would have been right for `make -j`; 56 is right for a
//! JIT, and they are not the same 4.
//!
//! **4. "`NUM_MMA_KV` must be instantiated four ways."** `kernels.def` does not
//! say this in as many words, but `DISPATCH_NUM_MMA_KV`
//! (`utils.cuh:116-133`) is a runtime switch over `{8, 4, 2, 1}` driven by a
//! `cudaDeviceGetAttribute`, so the archive had to emit all four. Under a JIT
//! [`crate::fa2::PrefillGeometry::derive`] computes the value in Rust from a
//! [`crate::fa2::Device`] and the fire names the ONE unit that holds it. The
//! rows for the others still exist — a row is three pointers — but their units
//! are never compiled on a part that does not select them. **This is the
//! largest single saving in the port and it is invisible in the row count.**
//!
//! # Why the rows are macro-generated and the constants are literals
//!
//! Every number in a `decode_unit!` or `prefill_unit!` invocation below is a
//! template argument of an upstream `__global__`, and
//! [`crate::fa2`] derives all of them from the same C++ lines. They are spelled
//! as literals here because a `DeviceKernel`'s two fields are `&'static str`
//! and there is no const string formatting — and the gap that opens between a
//! literal and a derivation is closed by [`tests`], which re-derives every one
//! and asserts equality. **A wrong literal is a failing unit test, not a wrong
//! kernel.**
//!
//! # What a unit is here
//!
//! * decode: one unit per **(head_dim, GQA group)** — 20 units, 5 rows each.
//!   Five rows because `dispatch_decode` and `dispatch_decode_capture`
//!   (`attention_flashinfer_common.cuh:686-753`) choose between five
//!   (variant, params) pairs from run-time flags a single fire can flip.
//! * prefill: one unit per **(head_dim, CTA_TILE_Q, NUM_MMA_KV)** — 36 units,
//!   10 rows each. Ten rows for the same reason, from
//!   `prefill`/`prefill_capture`/`prefill_custom` (`:775-857`).
//!
//! 36 and not 4x4x4=64 because `KernelTraits::IsInvalid()` (`prefill.cuh:221`)
//! prunes the rest; see [`prefill_unit!`] for the pruning, point by point.
//!
//! # What went, and what the list it went against was
//!
//! This file is north-star §5 step 8's device half, and the host half is now
//! written. `driver-cuda/csrc/attn/attention_flashinfer.cu` (1,258 lines) and
//! `plan_lifecycle.cpp` (105) are **deleted**, along with the whole of
//! `driver-cuda/csrc/`, which they were the last two files in. The measured
//! census that permitted it: `__global__` 0, `__device__` 0, and one real
//! `<<<>>>` — `attn_score_fold_heads` launching
//! `device::attn_score_fold_heads`, which is ours, already rowed, and already
//! fired from Rust by `driver-cuda/src/fire/attn_score.rs`. **The driver's
//! `<<<>>>` census is zero for the first time since 401.**
//!
//! Where the host half went:
//!
//! * `driver-cuda/src/fire/flashinfer_fa2.rs` — the two plan factories, the
//!   static non-split decode short-circuit, the descriptor H2D and the two
//!   `fire_*` entry points.
//! * `.../fire/flashinfer_fa2_dispatch.rs` — the four `switch
//!   (cache.head_dim)` dispatches, as symbol + grid + filled params.
//! * `.../bind/service.rs` — the six `RUST_SERVED` rows' bodies, which is
//!   where a module and a stream meet and the only place anything launches.
//! * `.../bind/mod.rs` — `DecodePlan`/`PrefillPlan`, now owning boxed Rust
//!   caches whose deleter is `Drop`.
//!
//! THE `build.rs` HUNKS, which this header named in advance so the deletion
//! would be a list and not a search. All were taken, all in
//! `crates/driver-cuda/build.rs`:
//!
//! * The `cc::Build` itself, `:672-743` — `fa2.cuda(true).std("c++17")`, the
//!   include list, `-gencode arch=compute_89,code=sm_89`, `--extended-lambda`,
//!   `--expt-relaxed-constexpr`, the `csrc/attn` directory scan, the
//!   `attn_units > 0` assertion at `:738` and `fa2.compile(...)` at `:743`.
//!   **Gone.** It was the last `.cuda(true)` in that script, so **nvcc is
//!   zero as well**.
//! * `DEP_PIE_KERNELS_CUDA_FLASHINFER` at `:664` and its CCCL sibling — the
//!   two `expect`s that read the vendored trees out of the kernels crate's
//!   metadata. **Gone.** Under NVRTC those trees reach a compile as carried
//!   headers and nothing consumes the env vars.
//! * **`println!("cargo:rustc-link-search=native=...")` and
//!   `println!("cargo:rustc-link-lib=static=pie_attn_flashinfer")`, `:800-801`.**
//!   **Gone**, and worth having been listed, because the general rule points
//!   the other way: `cc::Build::compile()` emits its own `rustc-link-lib` and
//!   leaves nothing to delete. This build was one of the two exceptions — it
//!   called `.cargo_metadata(false)` at `:703` and hand-printed the pair — so
//!   both lines were real deletions, and leaving them behind would have been
//!   a link error naming a library nothing builds any more.
//! * The four-point "why nvcc and not NVRTC" comment block above the build.
//!   **Deleted rather than defeated.** Points 1, 3 and 4 were claims about
//!   `attention_flashinfer.cu` in particular and die with it. Point 2's price
//!   — §13.6's FlashInfer patch set plus the bit-exact intrinsics — is
//!   repointed by name at **FA3, the SM90 Hopper prefill, whose headers are
//!   CPM-only**: the internalised closure has no `attention/hopper/` at all,
//!   so NVRTC
//!   cannot see them and this port is not evidence that the next one is
//!   cheap.
//!
//!   **CORRECTION, and it was mine: that sentence said "FA3 **and MLA**", and
//!   MLA does not belong in it.** The reason given covers FA3 only.
//!   `csrc/src/attn/flashinfer/attention/` carries `mla.cuh` (54 KB),
//!   `mla_params.cuh`, `scheduler.cuh` (87 KB) and `fastdiv.cuh` — which is
//!   every FlashInfer header `kernels-cuda/csrc/src/attn/attention_mla.cu`
//!   includes, measured against its include list rather than assumed.
//!   `table::attn`'s MLA block has always stated this correctly (*"THE HEADER
//!   GATE: this row clears it"*); this file was the one that was wrong, and
//!   at least one later pass inherited the error from here and had to
//!   re-measure to find it. What actually blocks MLA is one level down and
//!   nothing to do with headers: its FA2 arm passes an `MLAParams` BY VALUE,
//!   which wants `ArgValue::Bytes`, which only `x::Abi` produces — the same
//!   capability XQA's `KVCacheList` waits on.
//!
//!   The general lesson is worth more than the correction: a claim of the
//!   form "X and Y are blocked, because <reason>" has to have the reason
//!   checked against X and against Y SEPARATELY, because a shared verdict
//!   hides an unshared cause and reads as evidence for both.
//! * A fourth item the list did not have: `archive_src`, `jit_headers` and
//!   `jit_shims`, the three locals a comment kept alive with *"because the
//!   FA2 block below uses all three"*, plus the `jit_shims/cuda_fp16.h`
//!   assertion that guarded a C++ TU this crate no longer has.
//!
//! And in the tree, all taken: the seven `pie_x_*` declarations in
//! `bind/abi.rs` and their entry in `scripts/csrc-reachability-audit.py`'s
//! `DIRECT_ROOTS`; the four `dequant_kv_cache_layer_to_bf16_active` callers
//! at `attention_flashinfer.cu:648`, `:675`, `:1098`, `:1244`, which were the
//! last thing holding the census above zero and are now four calls to
//! `driver-cuda/src/fire/kv_paged.rs`'s Rust replacement; and the sweep of
//! `model-compiler/src/dsl.rs`, `lower.rs::semantic()` and
//! `crates/model/src` for **both** the symbol string and the DSL wrapper name
//! — different tokens, and a sweep for one of them once reported a live
//! symbol as uncalled. That sweep found all six rows live and none deleted:
//! the rows stayed, only their language changed.
//!
//! # THE HAZARD THIS CLOSED
//!
//! `kernels-cuda-new/csrc/shim/cuda/cmath:245-280` records that this shim's
//! `__fast_div_modulo` is `{u32 @0, u64 @8}` align 8 while **CCCL's** is
//! `{u32, u32, u32, i32}` align 4 — so `paged_kv_t::num_heads` sat at **+24
//! under the shim and +20 under CCCL**, with `sizeof` reconverging at 96
//! under both. `fa2/params.rs`' mirror is pinned to the SHIM's layout, which
//! is correct for every JIT fire; `attention_flashinfer.cu` compiled against
//! real CCCL and filled the **+20** layout. Both were correct, and a params
//! block filled on one side and read on the other is a silent wrong answer
//! rather than a crash. It was safe only because neither read the other's.
//! With the `cc::Build` gone there is one layout in the process and the
//! question cannot be asked again — which is why 4 and 5 belonged in the same
//! pass as the seams and not after them.

use crate::unit::Unit;

/// `kernels.def`'s `PIE_ATTN_HEAD_DIM` list, in its order.
///
/// Read by [`tests`] to check that the unit list covers exactly this set, so
/// that adding a point is one edit here plus its units and forgetting the
/// units is a test failure rather than a `Refusal` at some checkpoint's first
/// token.
pub const HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

/// `kernels.def`'s `PIE_ATTN_DECODE_GQA` list, in its order.
///
/// `DISPATCH_GQA_GROUP_SIZE` (`utils.cuh:164-183`) accepts exactly this set,
/// so it is upstream's list as much as ours. 5, 6 and 7 are absent because
/// `force_prefill_path` routes them; see this module's header.
pub const DECODE_GQA: &[u32] = &[1, 2, 3, 4, 8];

/// `PagedTraits` and the six variant aliases live in one header, and every unit
/// in this file compiles that one header.
///
/// A `&'static str` bound once rather than 56 `include_str!` invocations of the
/// same file: `include_str!` is expanded per call site, so 56 of them is 56
/// copies of the text in the binary.
const ROOT: &str = include_str!("../../csrc/src/attn/fa2.cuh");

/// `--device-as-default-execution-space`, and it is load-bearing.
///
/// Without it NVRTC rejects FlashInfer's `Dispatched` launcher templates with
/// *"A function without execution space annotations is considered a `__host__`
/// function"* — twelve errors, all in host code this crate never calls, all of
/// which vanish when the flag is on. Measured against `libnvrtc.so.13` on this
/// box before any of this file was written; [`crate::unit::Unit::options`]'s
/// own doc names FlashInfer as the case that field exists for.
///
/// # One flag, and nothing else — in particular no `-I`
///
/// This list is APPENDED to `runtime::nvrtc::options` (`nvrtc.rs:861`), which
/// is `--gpu-architecture=sm_XY -std=c++17 --fmad=false --prec-div=true
/// --prec-sqrt=true` and is the whole shared contract. NVRTC reads the list in
/// order and a later flag wins, so a unit can only ADD to that contract or
/// override it deliberately, and `Unit::cache_key` spans the same strings — an
/// override cannot be served a cubin built without it. One flag here is the
/// claim that FA2 needs nothing from the contract changed.
///
/// **Do not put an include path here.** The FA2 lattice was derived with a
/// hand-run `libnvrtc` probe that passed `-I csrc/src -I csrc/shim
/// -I csrc/vendor -I /usr/local/cuda/include` — a third root that no longer
/// exists, since the closure is under `csrc/src/attn/` now and the first root
/// already covers it — and that probe is a faithful
/// simulation of the fire, not the fire. This crate's NVRTC **passes no `-I`
/// at all and reads nothing from disk**: headers arrive as `includeNames[]`
/// from the carried set (`carried.rs`, generated by walking `csrc/`), which is
/// why the three probe roots resolve the same names — the directory layout was
/// chosen to make that true. An `-I` written here would name a path that does
/// not have to exist on the machine that fires.
///
/// The three numerics flags are worth naming for a second reason: they are why
/// bit-exactness against the AOT build is even discussable, and they are
/// already shared. FA2 must not restate them — a unit that repeated
/// `--fmad=false` would be indistinguishable in the cache key from one that
/// meant to override it.
const OPTIONS: &[&str] = &["--device-as-default-execution-space"];

/// One decode unit: five rows over one `(head_dim, GQA group)` point.
///
/// The six template constants are [`crate::fa2::DecodeGeometry`]'s, in
/// `BatchDecodeWithPagedKVCacheKernel`'s parameter order
/// (`decode.cuh:618-621`):
///
/// ```text
/// PosEncodingMode, num_stages_smem, tile_size_per_bdx, vec_size, bdx, bdy, bdz,
/// AttentionVariant, Params
/// ```
///
/// `PosEncodingMode` is `kNone` for every row — pie applies RoPE before
/// attention, so `POS_ENC` is fixed in `fa2.cuh` and never an axis.
///
/// # The five rows
///
/// `dispatch_decode` (`attention_flashinfer_common.cuh:686-720`) picks three:
/// `full` when the layer is full-attention and neither a window nor a soft cap
/// is asked for, `softcap` when a soft cap is (note it is the WINDOWED softcap
/// variant even for a full-attention layer — upstream's arm order, transcribed
/// rather than tidied), and `window` otherwise.
///
/// `dispatch_decode_capture` (`:722-753`) picks two more, over the capture
/// params, and refuses soft-cap and window outright — the alias comment at
/// `:129-142` gives both reasons and neither is a gap.
macro_rules! decode_unit {
    (
        $unit:ident, hd = $hd:literal, gqa = $g:literal,
        stages = $ns:literal, tile = $tile:literal, vec = $vec:literal,
        bdx = $bdx:literal, bdy = $bdy:literal, bdz = $bdz:literal,
        $(#[$note:meta])*
    ) => {
        $(#[$note])*
        pub mod $unit {
            use kernels::{KernelSig, kernel};

            use super::{OPTIONS, ROOT};
            use crate::device::DeviceKernel;
            use crate::unit::Unit;

            /// The head dim and GQA group this unit is the lattice point for.
            pub const POINT: (u32, u32) = ($hd, $g);

            const PATH: &str = "::flashinfer::BatchDecodeWithPagedKVCacheKernel";

            #[rustfmt::skip]
            static SIGS: [KernelSig; 5] = [
                // NO `operands`, and that is the third of the three ways a row
                // loses its `emit_c_shim` entry (`device::JIT_DISPATCHED` and
                // `execution::RUST_SERVED` are the other two).
                //
                // It is not a gap. The `__global__` takes ONE argument -- a
                // `__grid_constant__ BatchDecodeParams` passed by value
                // (`decode.cuh:621`) -- an aggregate of pointers, strides, a
                // `paged_kv_t` and eight scalars. `kernels::Ty` has no variant
                // for it and should not grow one: a `Ty` is a thing
                // `Args::bind` can CHECK, and an opaque blob is exactly the
                // thing it cannot. The fire builds the struct and launches it
                // through the raw path, which is why every row here is also
                // `LaunchRule::Unstated` -- the geometry is
                // `crate::fa2::DecodeGeometry`'s, not a rule's.
                kernel!(decode_full
                    concat!("attn::fa2::decode_hd", stringify!($hd), "_g", stringify!($g), "_full"),
                    file = Some("attn/fa2.cuh")),
                kernel!(decode_softcap
                    concat!("attn::fa2::decode_hd", stringify!($hd), "_g", stringify!($g), "_softcap"),
                    file = Some("attn/fa2.cuh")),
                kernel!(decode_window
                    concat!("attn::fa2::decode_hd", stringify!($hd), "_g", stringify!($g), "_window"),
                    file = Some("attn/fa2.cuh")),
                kernel!(decode_capture_full
                    concat!("attn::fa2::decode_hd", stringify!($hd), "_g", stringify!($g), "_capture_full"),
                    file = Some("attn/fa2.cuh")),
                kernel!(decode_capture_window
                    concat!("attn::fa2::decode_hd", stringify!($hd), "_g", stringify!($g), "_capture_window"),
                    file = Some("attn/fa2.cuh")),
            ];

            #[rustfmt::skip]
            static ROWS: [DeviceKernel; 5] = [
                DeviceKernel { sig: &SIGS[0], template_path: PATH, elem: concat!(
                    "::flashinfer::PosEncodingMode::kNone, ",
                    stringify!($ns), ", ", stringify!($tile), ", ", stringify!($vec), ", ",
                    stringify!($bdx), ", ", stringify!($bdy), ", ", stringify!($bdz), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantFull, ",
                    "::pie_cuda_driver::kernels::attn::fa2::DecodeParams") },
                DeviceKernel { sig: &SIGS[1], template_path: PATH, elem: concat!(
                    "::flashinfer::PosEncodingMode::kNone, ",
                    stringify!($ns), ", ", stringify!($tile), ", ", stringify!($vec), ", ",
                    stringify!($bdx), ", ", stringify!($bdy), ", ", stringify!($bdz), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantWindowSoftcap, ",
                    "::pie_cuda_driver::kernels::attn::fa2::DecodeParams") },
                DeviceKernel { sig: &SIGS[2], template_path: PATH, elem: concat!(
                    "::flashinfer::PosEncodingMode::kNone, ",
                    stringify!($ns), ", ", stringify!($tile), ", ", stringify!($vec), ", ",
                    stringify!($bdx), ", ", stringify!($bdy), ", ", stringify!($bdz), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantWindow, ",
                    "::pie_cuda_driver::kernels::attn::fa2::DecodeParams") },
                DeviceKernel { sig: &SIGS[3], template_path: PATH, elem: concat!(
                    "::flashinfer::PosEncodingMode::kNone, ",
                    stringify!($ns), ", ", stringify!($tile), ", ", stringify!($vec), ", ",
                    stringify!($bdx), ", ", stringify!($bdy), ", ", stringify!($bdz), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::CaptureFull, ",
                    "::pie_cuda_driver::kernels::attn::fa2::DecodeCaptureParams") },
                DeviceKernel { sig: &SIGS[4], template_path: PATH, elem: concat!(
                    "::flashinfer::PosEncodingMode::kNone, ",
                    stringify!($ns), ", ", stringify!($tile), ", ", stringify!($vec), ", ",
                    stringify!($bdx), ", ", stringify!($bdy), ", ", stringify!($bdz), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::CaptureWindow, ",
                    "::pie_cuda_driver::kernels::attn::fa2::DecodeCaptureParams") },
            ];

            /// The unit itself.
            pub const UNIT: Unit = Unit {
                name: concat!(
                    "attn/fa2_decode_hd", stringify!($hd), "_g", stringify!($g)
                ),
                root: ROOT,
                rows: &ROWS,
                options: OPTIONS,
            };
        }
    };
}

/// One prefill unit: ten rows over one `(head_dim, CTA_TILE_Q, NUM_MMA_KV)`
/// point.
///
/// `BatchPrefillWithPagedKVCacheKernel` takes TWO template arguments
/// (`prefill.cuh:3966-3967`) — a `KernelTraits` and a `Params` — and
/// `KernelTraits` takes fifteen (`:154-159`). `fa2.cuh`'s `PagedTraits` alias
/// supplies the five invariant types and `kNone`, so a row states the mask, the
/// seven integers and the variant.
///
/// # Which points exist, and why the other 28 do not
///
/// `KernelTraits::IsInvalid()` (`prefill.cuh:221-232`) prunes them, and every
/// clause that fires for this lattice is arithmetic on the numbers below —
/// `DTypeQKAccum` is `float`, `POS_ENCODING_MODE` is `kNone` and `DTypeKV` is
/// 2 bytes, which makes three of the six clauses unreachable. What is left:
///
/// * **`HEAD_DIM_VO >= 512 ? CTA_TILE_Q > 32 : CTA_TILE_Q == 32`** — upstream's
///   own comment calls this *"pairs `FA2DetermineCtaTileQ` never selects"*. It
///   gives head dims 64/128/256 the tiles `{16, 64, 128}` and head dim 512 the
///   tiles `{16, 32}`.
/// * **`NUM_MMA_D_VO == 4 && NUM_MMA_KV % 2 == 1`** — at head dim 64 only,
///   which drops `NUM_MMA_KV = 1` there.
/// * **`NUM_MMA_Q * (8 * NUM_MMA_D_VO_TILE + 8 * NUM_MMA_KV) >= 256`** — the
///   register-file bound, and the one that bites:
///   * head dim 128 at `CTA_TILE_Q = 128` (`NUM_MMA_Q = 2`) drops
///     `NUM_MMA_KV = 8`;
///   * **head dim 256 at `CTA_TILE_Q = 128` is invalid for every
///     `NUM_MMA_KV`** — `2 * (8*16 + 8*kv) >= 256` holds at `kv = 0`. There is
///     no valid instantiation of that pair at all, which is a fact about
///     upstream and not about this port. See [`tests`] and this file's report:
///     if `crate::plan::arith::fa2_determine_cta_tile_q` can return 128 for a
///     head dim of 256, the ARCHIVE reached `FLASHINFER_ERROR` there too.
///
/// # The ten rows
///
/// `prefill` (`attention_flashinfer_common.cuh:775-805`) picks six:
/// `{kCausal, kNone} x {FullSoftcap, Full}` for a full-attention layer, and
/// `kCausal x {WindowSoftcap, Window}` for a windowed one — upstream's
/// asymmetry, not a transcription slip: a windowed prefill is always causal
/// here because the only non-causal caller is the full-attention branch.
/// `prefill_capture` (`:806-837`) picks two, `prefill_custom` (`:838-857`) two
/// more over `kCustom`.
macro_rules! prefill_unit {
    (
        $unit:ident, hd = $hd:literal, q = $q:literal, kv = $kv:literal,
        mma_q = $mmaq:literal, d_qk = $dqk:literal, d_vo = $dvo:literal,
        warps_q = $wq:literal, warps_kv = $wkv:literal,
        $(#[$note:meta])*
    ) => {
        $(#[$note])*
        pub mod $unit {
            use kernels::{KernelSig, kernel};

            use super::{OPTIONS, ROOT};
            use crate::device::DeviceKernel;
            use crate::unit::Unit;

            /// The head dim, `CTA_TILE_Q` and `NUM_MMA_KV` this unit is the
            /// lattice point for.
            pub const POINT: (u32, u32, u32) = ($hd, $q, $kv);

            const PATH: &str = "::flashinfer::BatchPrefillWithPagedKVCacheKernel";

            #[rustfmt::skip]
            static SIGS: [KernelSig; 10] = [
                // No `operands`, for the reason `decode_unit!`'s rows give:
                // the `__global__` takes one `BatchPrefillPagedParams` by
                // value and `Args::bind` has nothing to check it with.
                kernel!(prefill_causal_full_softcap concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_causal_full_softcap"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_none_full_softcap concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_none_full_softcap"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_causal_full concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_causal_full"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_none_full concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_none_full"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_causal_softcap concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_causal_softcap"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_causal_window concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_causal_window"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_causal_capture concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_causal_capture"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_none_capture concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_none_capture"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_custom_softcap concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_custom_softcap"), file = Some("attn/fa2.cuh")),
                kernel!(prefill_custom concat!("attn::fa2::prefill_hd",
                    stringify!($hd), "_q", stringify!($q), "_kv", stringify!($kv),
                    "_custom"), file = Some("attn/fa2.cuh")),
            ];

            // `PagedTraits<MASK, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV,
            // NUM_MMA_D_QK, NUM_MMA_D_VO, NUM_WARPS_Q, NUM_WARPS_KV, Variant>`
            // then the params type -- the two arguments
            // `BatchPrefillWithPagedKVCacheKernel` takes
            // (`prefill.cuh:3966-3967`).
            #[rustfmt::skip]
            static ROWS: [DeviceKernel; 10] = [
                DeviceKernel { sig: &SIGS[0], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantFullSoftcap>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[1], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kNone, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantFullSoftcap>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[2], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantFull>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[3], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kNone, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantFull>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[4], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantWindowSoftcap>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[5], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantWindow>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[6], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCausal, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::CapturePrefill>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillCaptureParams") },
                DeviceKernel { sig: &SIGS[7], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kNone, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::CapturePrefill>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillCaptureParams") },
                DeviceKernel { sig: &SIGS[8], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCustom, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantCustomSoftcap>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
                DeviceKernel { sig: &SIGS[9], template_path: PATH, elem: concat!(
                    "::pie_cuda_driver::kernels::attn::fa2::PagedTraits<::flashinfer::MaskMode::kCustom, ",
                    stringify!($q), ", ", stringify!($mmaq), ", ", stringify!($kv), ", ",
                    stringify!($dqk), ", ", stringify!($dvo), ", ",
                    stringify!($wq), ", ", stringify!($wkv), ", ",
                    "::pie_cuda_driver::kernels::attn::fa2::VariantCustom>, ",
                    "::pie_cuda_driver::kernels::attn::fa2::PrefillParams") },
            ];

            /// The unit itself.
            pub const UNIT: Unit = Unit {
                name: concat!(
                    "attn/fa2_prefill_hd", stringify!($hd),
                    "_q", stringify!($q), "_kv", stringify!($kv)
                ),
                root: ROOT,
                rows: &ROWS,
                options: OPTIONS,
            };
        }
    };
}

// ─── the decode lattice: 4 head dims x 5 GQA groups ─────────────────────────
//
// Every constant is `crate::fa2::DecodeGeometry::derive(hd, g, KvWidth::BF16,
// Device { cc_major: 8, .. })`, and `tests::decode_literals_match_the_derivation`
// re-derives all 120 of them. `stages = 2` on every row is
// `DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM` at compute capability >= 8
// (`utils.cuh:349-356`) -- an sm_89-only lattice, exactly as the archive's
// `-gencode arch=compute_89,code=sm_89` was, and for the same reason.
//
// Note `bdz` is an INTEGER division (`decode.cuh:769`), which is why GQA 3
// gives blocks of 120, 96 and 96 threads rather than 128.

decode_unit!(d_hd64_g1, hd = 64, gqa = 1, stages = 2, tile = 4, vec = 8, bdx = 8, bdy = 1, bdz = 16,
    /// llama3.2-1B, MQA. `tile_size_per_bdx = 4` is the GQA-1 special case
    /// (`decode.cuh:770`): one query head per KV head would leave a CTA four
    /// times too small, so the KV tile widens instead of the block. 36,864 B
    /// of shared memory, the largest of any GQA group at this head dim.
);
decode_unit!(d_hd64_g2, hd = 64, gqa = 2, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 2, bdz = 8,);
decode_unit!(d_hd64_g3, hd = 64, gqa = 3, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 3, bdz = 5,
    /// `128 / (8*3) = 5`, so the block is 8x3x5 = **120 threads**. Upstream
    /// launches the 120; see `crate::fa2::DecodeGeometry::bdz`.
);
decode_unit!(d_hd64_g4, hd = 64, gqa = 4, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 4, bdz = 4,);
decode_unit!(d_hd64_g8, hd = 64, gqa = 8, stages = 2, tile = 1, vec = 8, bdx = 8, bdy = 8, bdz = 2,);

decode_unit!(d_hd128_g1, hd = 128, gqa = 1, stages = 2, tile = 4, vec = 8, bdx = 16, bdy = 1, bdz = 8,);
decode_unit!(d_hd128_g2, hd = 128, gqa = 2, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 2, bdz = 4,);
decode_unit!(d_hd128_g3, hd = 128, gqa = 3, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 3, bdz = 2,);
decode_unit!(d_hd128_g4, hd = 128, gqa = 4, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 4, bdz = 2,
    /// qwen3 and qwen2's usual shape, and **the point the pre-port NVRTC probe
    /// guessed wrong**: it passed `tile = 4, bdz = 1`, which is a valid
    /// instantiation the launcher never selects — `tile_size_per_bdx = 4`
    /// requires `GROUP_SIZE == 1` (`decode.cuh:770`) and `bdz` here is
    /// `128 / (16*4) = 2`. The probe proved NVRTC compiles the template; it did
    /// not prove which specialisation the host picks, and the difference is
    /// this row.
);
decode_unit!(d_hd128_g8, hd = 128, gqa = 8, stages = 2, tile = 1, vec = 8, bdx = 16, bdy = 8, bdz = 1,);

decode_unit!(d_hd256_g1, hd = 256, gqa = 1, stages = 2, tile = 4, vec = 8, bdx = 32, bdy = 1, bdz = 4,);
decode_unit!(d_hd256_g2, hd = 256, gqa = 2, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 2, bdz = 2,);
decode_unit!(d_hd256_g3, hd = 256, gqa = 3, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 3, bdz = 1,);
decode_unit!(d_hd256_g4, hd = 256, gqa = 4, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 4, bdz = 1,);
decode_unit!(d_hd256_g8, hd = 256, gqa = 8, stages = 2, tile = 1, vec = 8, bdx = 32, bdy = 8, bdz = 1,
    /// The one decode point where `num_threads` exceeds 128: `bdx*bdy = 256`,
    /// so `max(128, 256) = 256` (`decode.cuh:768`) and the shared-memory tail
    /// doubles with it — 18,432 B against 9,216 at every other group here.
);

decode_unit!(d_hd512_g1, hd = 512, gqa = 1, stages = 2, tile = 4, vec = 16, bdx = 32, bdy = 1, bdz = 4,
    /// **69,632 B of dynamic shared memory** — over the 48 KB default cap, so
    /// this is the one FA2 row that needs
    /// `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)`.
    /// `KernelModule::fire` raises it automatically above
    /// `DEFAULT_DYNAMIC_SMEM`, so the fire states three fields and nothing
    /// else. `vec_size = 16` here and 8 everywhere else: `HEAD_DIM / 32` beats
    /// `16 / sizeof(DTypeKV)` only at 512 (`decode.cuh:762`).
);
decode_unit!(d_hd512_g2, hd = 512, gqa = 2, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 2, bdz = 2,);
decode_unit!(d_hd512_g3, hd = 512, gqa = 3, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 3, bdz = 1,);
decode_unit!(d_hd512_g4, hd = 512, gqa = 4, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 4, bdz = 1,);
decode_unit!(d_hd512_g8, hd = 512, gqa = 8, stages = 2, tile = 1, vec = 16, bdx = 32, bdy = 8, bdz = 1,);

// ─── the prefill lattice: 36 valid (head_dim, CTA_TILE_Q, NUM_MMA_KV) points ─
//
// `mma_q`, `warps_q` and `warps_kv` come from `get_num_mma_q`,
// `get_num_warps_q` and `get_num_warps_kv` (`prefill.cuh:72-96`) applied to
// `CTA_TILE_Q`, except at head dim 512 with `CTA_TILE_Q = 32`, where
// `kBf16VOSplit` (`:4191-4195`) overrides all three to `(1, 2, 2)`.
// `d_qk` and `d_vo` are `HEAD_DIM / 16` (`:4206-4207`).
//
// `NUM_MMA_KV` is the axis a JIT does not have to enumerate at fire time:
// `crate::fa2::PrefillGeometry::derive` picks ONE from the device's shared
// memory budget, and the other units in the same (head_dim, CTA_TILE_Q) column
// are never compiled on that part. They are declared because the value is a
// device fact and this table is not allowed to assume a card.

prefill_unit!(p_hd64_q16_kv8, hd = 64, q = 16, kv = 8, mma_q = 1, d_qk = 4, d_vo = 4, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd64_q16_kv4, hd = 64, q = 16, kv = 4, mma_q = 1, d_qk = 4, d_vo = 4, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd64_q16_kv2, hd = 64, q = 16, kv = 2, mma_q = 1, d_qk = 4, d_vo = 4, warps_q = 1, warps_kv = 4,
    /// `NUM_MMA_KV = 1` is absent at head dim 64 for every tile:
    /// `NUM_MMA_D_VO == 4 && NUM_MMA_KV % 2 == 1` is `IsInvalid()`'s second
    /// clause (`prefill.cuh:224`).
);
prefill_unit!(p_hd64_q64_kv8, hd = 64, q = 64, kv = 8, mma_q = 1, d_qk = 4, d_vo = 4, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd64_q64_kv4, hd = 64, q = 64, kv = 4, mma_q = 1, d_qk = 4, d_vo = 4, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd64_q64_kv2, hd = 64, q = 64, kv = 2, mma_q = 1, d_qk = 4, d_vo = 4, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd64_q128_kv8, hd = 64, q = 128, kv = 8, mma_q = 2, d_qk = 4, d_vo = 4, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd64_q128_kv4, hd = 64, q = 128, kv = 4, mma_q = 2, d_qk = 4, d_vo = 4, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd64_q128_kv2, hd = 64, q = 128, kv = 2, mma_q = 2, d_qk = 4, d_vo = 4, warps_q = 4, warps_kv = 1,);

prefill_unit!(p_hd128_q16_kv8, hd = 128, q = 16, kv = 8, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd128_q16_kv4, hd = 128, q = 16, kv = 4, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd128_q16_kv2, hd = 128, q = 16, kv = 2, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd128_q16_kv1, hd = 128, q = 16, kv = 1, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd128_q64_kv8, hd = 128, q = 64, kv = 8, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd128_q64_kv4, hd = 128, q = 64, kv = 4, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd128_q64_kv2, hd = 128, q = 64, kv = 2, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd128_q64_kv1, hd = 128, q = 64, kv = 1, mma_q = 1, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd128_q128_kv4, hd = 128, q = 128, kv = 4, mma_q = 2, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,
    /// `NUM_MMA_KV = 8` is absent here: `2 * (8*8 + 8*8) = 256`, which is
    /// `IsInvalid()`'s register clause exactly at the bound
    /// (`prefill.cuh:226-228`).
);
prefill_unit!(p_hd128_q128_kv2, hd = 128, q = 128, kv = 2, mma_q = 2, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd128_q128_kv1, hd = 128, q = 128, kv = 1, mma_q = 2, d_qk = 8, d_vo = 8, warps_q = 4, warps_kv = 1,);

prefill_unit!(p_hd256_q16_kv8, hd = 256, q = 16, kv = 8, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd256_q16_kv4, hd = 256, q = 16, kv = 4, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd256_q16_kv2, hd = 256, q = 16, kv = 2, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd256_q16_kv1, hd = 256, q = 16, kv = 1, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd256_q64_kv8, hd = 256, q = 64, kv = 8, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd256_q64_kv4, hd = 256, q = 64, kv = 4, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd256_q64_kv2, hd = 256, q = 64, kv = 2, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 4, warps_kv = 1,);
prefill_unit!(p_hd256_q64_kv1, hd = 256, q = 64, kv = 1, mma_q = 1, d_qk = 16, d_vo = 16, warps_q = 4, warps_kv = 1,
    /// **`CTA_TILE_Q = 128` has no valid point at head dim 256.**
    /// `NUM_MMA_Q = 2` and `NUM_MMA_D_VO_TILE = 16`, so `IsInvalid()`'s
    /// register clause is `2 * (128 + 8*NUM_MMA_KV) >= 256`, which holds for
    /// every `NUM_MMA_KV` including zero. gemma2/3/3n reach head dim 256, so if
    /// `crate::plan::arith::fa2_determine_cta_tile_q` ever returns 128 for
    /// them the archive hit `FLASHINFER_ERROR` at that fire too. This is a
    /// fact about upstream that the port surfaces; it is not a port defect.
);

prefill_unit!(p_hd512_q16_kv8, hd = 512, q = 16, kv = 8, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd512_q16_kv4, hd = 512, q = 16, kv = 4, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd512_q16_kv2, hd = 512, q = 16, kv = 2, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd512_q16_kv1, hd = 512, q = 16, kv = 1, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 1, warps_kv = 4,);
prefill_unit!(p_hd512_q32_kv8, hd = 512, q = 32, kv = 8, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 2, warps_kv = 2,
    /// `kBf16VOSplit` (`prefill.cuh:4191`): 16-bit KV, head dim >= 512 and
    /// `CTA_TILE_Q == 32` together override `(NUM_MMA_Q, NUM_WARPS_Q,
    /// NUM_WARPS_KV)` to `(1, 2, 2)`, which `get_num_*` would have made
    /// `(2, 1, 4)`. This is the only point in the lattice where the three
    /// helpers are not the answer.
);
prefill_unit!(p_hd512_q32_kv4, hd = 512, q = 32, kv = 4, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 2, warps_kv = 2,);
prefill_unit!(p_hd512_q32_kv2, hd = 512, q = 32, kv = 2, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 2, warps_kv = 2,);
prefill_unit!(p_hd512_q32_kv1, hd = 512, q = 32, kv = 1, mma_q = 1, d_qk = 32, d_vo = 32, warps_q = 2, warps_kv = 2,);

/// Every FA2 unit: 20 decode, 36 prefill.
///
/// Order is not semantic — a unit's position is its slot in the module cache —
/// but it is stable, which is what keeps a diff readable.
pub const UNITS: &[Unit] = &[
    d_hd64_g1::UNIT,
    d_hd64_g2::UNIT,
    d_hd64_g3::UNIT,
    d_hd64_g4::UNIT,
    d_hd64_g8::UNIT,
    d_hd128_g1::UNIT,
    d_hd128_g2::UNIT,
    d_hd128_g3::UNIT,
    d_hd128_g4::UNIT,
    d_hd128_g8::UNIT,
    d_hd256_g1::UNIT,
    d_hd256_g2::UNIT,
    d_hd256_g3::UNIT,
    d_hd256_g4::UNIT,
    d_hd256_g8::UNIT,
    d_hd512_g1::UNIT,
    d_hd512_g2::UNIT,
    d_hd512_g3::UNIT,
    d_hd512_g4::UNIT,
    d_hd512_g8::UNIT,
    p_hd64_q16_kv8::UNIT,
    p_hd64_q16_kv4::UNIT,
    p_hd64_q16_kv2::UNIT,
    p_hd64_q64_kv8::UNIT,
    p_hd64_q64_kv4::UNIT,
    p_hd64_q64_kv2::UNIT,
    p_hd64_q128_kv8::UNIT,
    p_hd64_q128_kv4::UNIT,
    p_hd64_q128_kv2::UNIT,
    p_hd128_q16_kv8::UNIT,
    p_hd128_q16_kv4::UNIT,
    p_hd128_q16_kv2::UNIT,
    p_hd128_q16_kv1::UNIT,
    p_hd128_q64_kv8::UNIT,
    p_hd128_q64_kv4::UNIT,
    p_hd128_q64_kv2::UNIT,
    p_hd128_q64_kv1::UNIT,
    p_hd128_q128_kv4::UNIT,
    p_hd128_q128_kv2::UNIT,
    p_hd128_q128_kv1::UNIT,
    p_hd256_q16_kv8::UNIT,
    p_hd256_q16_kv4::UNIT,
    p_hd256_q16_kv2::UNIT,
    p_hd256_q16_kv1::UNIT,
    p_hd256_q64_kv8::UNIT,
    p_hd256_q64_kv4::UNIT,
    p_hd256_q64_kv2::UNIT,
    p_hd256_q64_kv1::UNIT,
    p_hd512_q16_kv8::UNIT,
    p_hd512_q16_kv4::UNIT,
    p_hd512_q16_kv2::UNIT,
    p_hd512_q16_kv1::UNIT,
    p_hd512_q32_kv8::UNIT,
    p_hd512_q32_kv4::UNIT,
    p_hd512_q32_kv2::UNIT,
    p_hd512_q32_kv1::UNIT,
];

/// The unit that holds one decode lattice point, by name.
///
/// The fire's resolution step: it knows a head dim and a GQA group, and it
/// needs a `&'static str` to hand [`crate::unit::unit_of`]. A `match` over the
/// twenty points rather than a formatted string, because the name must be
/// `&'static` and because a point outside the lattice must be a REFUSAL and not
/// a lookup miss — [`crate::fa2::Refusal::DecodeGroupSize`] says which group
/// was asked for, and `unit_of` returning `None` would say only that some name
/// was absent.
#[must_use]
pub fn decode_unit_name(head_dim: u32, group_size: u32) -> Option<&'static str> {
    let unit = match (head_dim, group_size) {
        (64, 1) => d_hd64_g1::UNIT,
        (64, 2) => d_hd64_g2::UNIT,
        (64, 3) => d_hd64_g3::UNIT,
        (64, 4) => d_hd64_g4::UNIT,
        (64, 8) => d_hd64_g8::UNIT,
        (128, 1) => d_hd128_g1::UNIT,
        (128, 2) => d_hd128_g2::UNIT,
        (128, 3) => d_hd128_g3::UNIT,
        (128, 4) => d_hd128_g4::UNIT,
        (128, 8) => d_hd128_g8::UNIT,
        (256, 1) => d_hd256_g1::UNIT,
        (256, 2) => d_hd256_g2::UNIT,
        (256, 3) => d_hd256_g3::UNIT,
        (256, 4) => d_hd256_g4::UNIT,
        (256, 8) => d_hd256_g8::UNIT,
        (512, 1) => d_hd512_g1::UNIT,
        (512, 2) => d_hd512_g2::UNIT,
        (512, 3) => d_hd512_g3::UNIT,
        (512, 4) => d_hd512_g4::UNIT,
        (512, 8) => d_hd512_g8::UNIT,
        _ => return None,
    };
    Some(unit.name)
}

/// One decode row's symbol, by lattice point and arm.
///
/// The five arms are the five `dispatch_decode`/`dispatch_decode_capture`
/// branches, in [`DecodeArm`]'s order.
#[must_use]
pub fn decode_symbol(head_dim: u32, group_size: u32, arm: DecodeArm) -> Option<&'static str> {
    let name = decode_unit_name(head_dim, group_size)?;
    let unit = UNITS.iter().find(|unit| unit.name == name)?;
    unit.rows.get(arm as usize).map(|row| row.sig.symbol)
}

/// Which of `dispatch_decode`'s five branches a fire took.
///
/// Named rather than an index because the branch is chosen from three run-time
/// flags and the mapping is upstream's arm ORDER, which is load-bearing: a
/// full-attention layer WITH a soft cap takes `WindowSoftcap` and not
/// `FullSoftcap`, because `dispatch_decode`'s first arm tests
/// `logits_soft_cap <= 0.f` before its second tests the cap at all
/// (`attention_flashinfer_common.cuh:701-715`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeArm {
    /// `AttnVariantFull` — full attention, no window, no soft cap.
    Full = 0,
    /// `AttnVariantSoftcap` — a soft cap, windowed or not.
    Softcap = 1,
    /// `AttnVariant` — the sliding-window default.
    Window = 2,
    /// `AttnScoreCaptureFull` over `DecodeScoreParams`.
    CaptureFull = 3,
    /// `AttnScoreCapture` over `DecodeScoreParams`.
    CaptureWindow = 4,
}

/// The unit that holds one prefill lattice point, by name.
///
/// `num_mma_kv` is [`crate::fa2::PrefillGeometry::num_mma_kv`] — DERIVED, not
/// chosen: the fire asks `PrefillGeometry::derive` for it and hands the answer
/// here. That is the whole of the `DISPATCH_NUM_MMA_KV` switch
/// (`utils.cuh:116-133`) moved from the device build to the host.
#[must_use]
pub fn prefill_unit_name(head_dim: u32, cta_tile_q: u32, num_mma_kv: u32) -> Option<&'static str> {
    let unit = match (head_dim, cta_tile_q, num_mma_kv) {
        (64, 16, 8) => p_hd64_q16_kv8::UNIT,
        (64, 16, 4) => p_hd64_q16_kv4::UNIT,
        (64, 16, 2) => p_hd64_q16_kv2::UNIT,
        (64, 64, 8) => p_hd64_q64_kv8::UNIT,
        (64, 64, 4) => p_hd64_q64_kv4::UNIT,
        (64, 64, 2) => p_hd64_q64_kv2::UNIT,
        (64, 128, 8) => p_hd64_q128_kv8::UNIT,
        (64, 128, 4) => p_hd64_q128_kv4::UNIT,
        (64, 128, 2) => p_hd64_q128_kv2::UNIT,
        (128, 16, 8) => p_hd128_q16_kv8::UNIT,
        (128, 16, 4) => p_hd128_q16_kv4::UNIT,
        (128, 16, 2) => p_hd128_q16_kv2::UNIT,
        (128, 16, 1) => p_hd128_q16_kv1::UNIT,
        (128, 64, 8) => p_hd128_q64_kv8::UNIT,
        (128, 64, 4) => p_hd128_q64_kv4::UNIT,
        (128, 64, 2) => p_hd128_q64_kv2::UNIT,
        (128, 64, 1) => p_hd128_q64_kv1::UNIT,
        (128, 128, 4) => p_hd128_q128_kv4::UNIT,
        (128, 128, 2) => p_hd128_q128_kv2::UNIT,
        (128, 128, 1) => p_hd128_q128_kv1::UNIT,
        (256, 16, 8) => p_hd256_q16_kv8::UNIT,
        (256, 16, 4) => p_hd256_q16_kv4::UNIT,
        (256, 16, 2) => p_hd256_q16_kv2::UNIT,
        (256, 16, 1) => p_hd256_q16_kv1::UNIT,
        (256, 64, 8) => p_hd256_q64_kv8::UNIT,
        (256, 64, 4) => p_hd256_q64_kv4::UNIT,
        (256, 64, 2) => p_hd256_q64_kv2::UNIT,
        (256, 64, 1) => p_hd256_q64_kv1::UNIT,
        (512, 16, 8) => p_hd512_q16_kv8::UNIT,
        (512, 16, 4) => p_hd512_q16_kv4::UNIT,
        (512, 16, 2) => p_hd512_q16_kv2::UNIT,
        (512, 16, 1) => p_hd512_q16_kv1::UNIT,
        (512, 32, 8) => p_hd512_q32_kv8::UNIT,
        (512, 32, 4) => p_hd512_q32_kv4::UNIT,
        (512, 32, 2) => p_hd512_q32_kv2::UNIT,
        (512, 32, 1) => p_hd512_q32_kv1::UNIT,
        _ => return None,
    };
    Some(unit.name)
}

/// One prefill row's symbol, by lattice point and arm.
#[must_use]
pub fn prefill_symbol(
    head_dim: u32,
    cta_tile_q: u32,
    num_mma_kv: u32,
    arm: PrefillArm,
) -> Option<&'static str> {
    let name = prefill_unit_name(head_dim, cta_tile_q, num_mma_kv)?;
    let unit = UNITS.iter().find(|unit| unit.name == name)?;
    unit.rows.get(arm as usize).map(|row| row.sig.symbol)
}

/// Which of the ten prefill branches a fire took.
///
/// From `AttnHd<HD>::prefill` (`attention_flashinfer_common.cuh:775-805`),
/// `::prefill_capture` (`:806-837`) and `::prefill_custom` (`:838-857`), in
/// declaration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillArm {
    /// `kCausal`, `AttnVariantFullSoftcap`.
    CausalFullSoftcap = 0,
    /// `kNone`, `AttnVariantFullSoftcap`.
    NoneFullSoftcap = 1,
    /// `kCausal`, `AttnVariantFull`.
    CausalFull = 2,
    /// `kNone`, `AttnVariantFull`.
    NoneFull = 3,
    /// `kCausal`, `AttnVariantSoftcap` — the windowed soft-cap variant, and
    /// causal only: `prefill`'s windowed branch has no `kNone` arm.
    CausalSoftcap = 4,
    /// `kCausal`, `AttnVariant`.
    CausalWindow = 5,
    /// `kCausal`, `AttnScoreCapturePrefill` over `PrefillScoreParams`.
    CausalCapture = 6,
    /// `kNone`, `AttnScoreCapturePrefill` over `PrefillScoreParams`.
    NoneCapture = 7,
    /// `kCustom`, `AttnVariantCustomSoftcap`.
    CustomSoftcap = 8,
    /// `kCustom`, `AttnVariantCustom`.
    Custom = 9,
}

#[cfg(test)]
mod tests {
    use super::{DECODE_GQA, HEAD_DIMS, UNITS, decode_unit_name, prefill_unit_name};
    use crate::fa2::{Device, KvWidth, PrefillGeometry};

    /// Every decode row's six template constants are the ones
    /// [`crate::fa2::DecodeGeometry::derive`] computes.
    ///
    /// **This is the join between the literal table and the derivation**, and
    /// the reason the literals are allowed to be literals. It reads the `elem`
    /// string back — the same string `nvrtcAddNameExpression` is handed — so a
    /// typo inside the macro's `concat!` is caught here and not by NVRTC.
    #[test]
    fn decode_literals_match_the_derivation() {
        for &head_dim in HEAD_DIMS {
            for &group in DECODE_GQA {
                let name = decode_unit_name(head_dim, group)
                    .unwrap_or_else(|| panic!("no unit for hd {head_dim} gqa {group}"));
                let unit = UNITS.iter().find(|unit| unit.name == name).unwrap();
                let geometry = crate::fa2::DecodeGeometry::derive(
                    head_dim,
                    group,
                    KvWidth::BF16,
                    Device::L40S,
                )
                .unwrap_or_else(|why| panic!("hd {head_dim} gqa {group}: {why}"));
                let wanted = format!(
                    "::flashinfer::PosEncodingMode::kNone, {}, {}, {}, {}, {}, {}, ",
                    geometry.num_stages_smem,
                    geometry.tile_size_per_bdx,
                    geometry.vec_size,
                    geometry.bdx,
                    geometry.bdy,
                    geometry.bdz,
                );
                for row in unit.rows {
                    assert!(
                        row.elem.starts_with(&wanted),
                        "hd {head_dim} gqa {group}: row states\n  {}\nderivation wants\n  {wanted}",
                        row.elem,
                    );
                }
            }
        }
    }

    /// Every prefill row's `KernelTraits` arguments are the ones
    /// [`PrefillGeometry`] computes for that point.
    ///
    /// `NUM_MMA_KV` is taken FROM the unit rather than derived, because it is
    /// the one argument that depends on the part; the other five are checked
    /// against the derivation, and the derivation's own `num_mma_kv` is checked
    /// separately by [`the_derived_num_mma_kv_names_a_unit`].
    #[test]
    fn prefill_literals_match_the_derivation() {
        for unit in UNITS.iter().filter(|unit| unit.name.contains("fa2_prefill")) {
            let point = unit
                .name
                .trim_start_matches("attn/fa2_prefill_hd")
                .split(['_'])
                .collect::<Vec<_>>();
            let head_dim: u32 = point[0].parse().unwrap();
            let cta_tile_q: u32 = point[1].trim_start_matches('q').parse().unwrap();
            let num_mma_kv: u32 = point[2].trim_start_matches("kv").parse().unwrap();
            let geometry =
                PrefillGeometry::derive(head_dim, cta_tile_q, KvWidth::BF16, true, Device::L40S)
                    .unwrap_or_else(|why| panic!("{}: {why}", unit.name));
            let wanted = format!(
                ", {cta_tile_q}, {}, {num_mma_kv}, {}, {}, {}, {}, ",
                geometry.num_mma_q,
                geometry.num_mma_d_qk,
                geometry.num_mma_d_vo,
                geometry.num_warps_q,
                geometry.num_warps_kv,
            );
            for row in unit.rows {
                assert!(
                    row.elem.contains(&wanted),
                    "{}: row states\n  {}\nderivation wants\n  {wanted}",
                    unit.name,
                    row.elem,
                );
            }
        }
    }

    /// The `NUM_MMA_KV` the derivation picks on this box names a unit that
    /// exists.
    ///
    /// The one check that would have caught a lattice pruned one point too far:
    /// a fire derives a value and then asks for a unit by name, and a value
    /// with no unit is an unfireable point rather than a compile error.
    ///
    /// `Device::L40S` and not a device query — this is layer 2 and runs with no
    /// GPU. A part with a different shared-memory budget picks a different
    /// point, which is exactly why all four are declared.
    #[test]
    fn the_derived_num_mma_kv_names_a_unit() {
        for &head_dim in HEAD_DIMS {
            for &cta_tile_q in &[16u32, 32, 64, 128] {
                let Ok(geometry) = PrefillGeometry::derive(
                    head_dim,
                    cta_tile_q,
                    KvWidth::BF16,
                    true,
                    Device::L40S,
                ) else {
                    continue; // upstream prunes this pair; see `prefill_unit!`
                };
                assert!(
                    prefill_unit_name(head_dim, cta_tile_q, geometry.num_mma_kv).is_some(),
                    "hd {head_dim} q {cta_tile_q} derived NUM_MMA_KV {} and no unit holds it",
                    geometry.num_mma_kv,
                );
            }
        }
    }

    /// No two FA2 units share a name and no two rows share a symbol.
    ///
    /// `unit_of` scans and answers with the first match, so a duplicate is
    /// unresolvable by construction rather than merely confusing.
    #[test]
    fn names_and_symbols_are_unique() {
        let mut names: Vec<&str> = Vec::new();
        let mut symbols: Vec<&str> = Vec::new();
        for unit in UNITS {
            assert!(!names.contains(&unit.name), "{} is declared twice", unit.name);
            names.push(unit.name);
            for row in unit.rows {
                assert!(
                    !symbols.contains(&row.sig.symbol),
                    "{} is stated twice",
                    row.sig.symbol
                );
                symbols.push(row.sig.symbol);
            }
        }
        assert_eq!(UNITS.len(), 56);
        assert_eq!(symbols.len(), 20 * 5 + 36 * 10);
    }

    /// Every FA2 row spells an ABSOLUTE instantiation.
    ///
    /// `DeviceKernel::instantiation`'s escape is what makes these rows
    /// possible; a row here that forgot the leading `::` would be prefixed with
    /// `::pie_cuda_driver::kernels::` and name nothing.
    #[test]
    fn every_row_is_absolutely_qualified() {
        for unit in UNITS {
            for row in unit.rows {
                let name = row.instantiation();
                assert!(
                    name.starts_with("::flashinfer::"),
                    "{} instantiates {name}",
                    row.sig.symbol
                );
                assert!(
                    !name.contains("::pie_cuda_driver::kernels::::"),
                    "{} double-qualified: {name}",
                    row.sig.symbol
                );
            }
        }
    }
}
