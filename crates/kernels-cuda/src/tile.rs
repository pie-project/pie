//! The CuTile roots, and the six things between them and a cubin.
//!
//! Twelve `.cuh` under `kernels/` are CuTile: eleven `__tile_global__` kernels
//! and `tile_alternatives.cuh`, the assertions-only translation unit that pins
//! each kernel's `*_tile_preferred` bound to the sweep row that produced it.
//! Five of them are program sources — the other seven are reached by
//! `tile_alternatives.cuh`'s own quoted `#include`s — and this module is where
//! those five are declared.
//!
//! **Not a family.** No [`Family`](crate::jit::Family) here, no routine, no
//! `mod inst`: nothing in this crate compiles a symbol out of any of them, and
//! the rest of this comment is the measured account of why. They are grouped
//! by that fact rather than filed with `sample`, `layout` and `quant`, because
//! that fact is the only thing about them a reader needs first and it is worth
//! exactly one statement. A `ROOT` sitting among `quant`'s live routines reads
//! as one more thing that fires.
//!
//! # Why they are roots at all, when nothing compiles them
//!
//! `src/source.rs` carries every file under `kernels/`, so carried is a superset of
//! compiled-against, and a `.cuh` in the gap is bytes in every process, a term
//! in every cache key, and — the expensive one — text a reader takes for a
//! kernel some routine fires. A [`Root`](crate::jit::Root) is this crate's
//! only way of saying that a file is a program source, so until these existed
//! the twelve had no statement about them anywhere in `src/` and read as
//! orphans.
//!
//! Declaring them says the true thing and no more: these are the texts a
//! compile of the tile alternatives is handed, and this crate has none.
//! `every_instantiation_compiles`'s `NO_INSTANTIATIONS` carries the other half
//! — that a root naming no instantiation is a defect unless it is written
//! down — so the five are named there too, and a sixth cannot arrive quietly.
//!
//! # Six things, five of them in this crate
//!
//! `tile_alternatives.cuh`'s header states that the alternatives need NVRTC
//! 13.3, 13.3 runtime headers and `tileiras`. That is true and it is not the
//! whole list. Measured on this box on 2026-08-16 — an L40S, by compiling the
//! text rather than by reading for it — a compile hits these in this order:
//!
//! 1. **The NVRTC that loads is 13.0.88.** `nvrtcVersion` answers 13.0 in
//!    process and `libnvrtc.so` resolves through `/usr/local/cuda` to
//!    `libnvrtc.so.13.0.88`, so the header's claim holds as written. The floor
//!    is not a version-number nicety: 13.0's own `crt/cuda_tile.h` is sixty
//!    lines declaring `cuda::cutile::print` and nothing else, against 4,064 in
//!    13.3. There are no tile types in 13.0 to compile against.
//!
//! 2. **`crt/cuda_tile.h` resolves under neither**, and this is the one that
//!    surprises. NVRTC gets no `-I` from this crate by design — an `#include`
//!    resolves against the carried set or it does not resolve — and 13.3 does
//!    not bundle this header either, so both answer `catastrophic error: could
//!    not open source file "crt/cuda_tile.h" (no directories in search list)`.
//!    It is NVIDIA's file, which is what `shim/`'s rule covers and what
//!    this crate exists not to need at build time. `sample/argmax_tile.cuh`
//!    spells it `<cuda_tile.h>`, which the 13.3 runtime ships as a separate
//!    file whose only directive includes the other, so the two spellings want
//!    two files supplied and not one.
//!
//! 3. **`-std=c++17` is hard-coded** for every compile in the crate, and
//!    `crt/cuda_tile.h:55` is an `#error` below C++20. A later `-std=c++20`
//!    does win — measured, over an `nvrtc: warning: "--std (-std)=c++17"
//!    followed by "--std (-std)=c++20"`.
//!
//! 4. **`-enable-tile` and `-default-device` are not passed.** The first is
//!    the dangerous one, because its absence is a WARNING and not an error:
//!    *"parsing for Tile constructs is not enabled. Tile annotations
//!    (including `__tile__` and `__tile_global__`) will be ignored"*. The
//!    compile then succeeds and has produced nothing. Without the second,
//!    every unannotated `__tile__` helper in NVIDIA's header is a host
//!    function in JIT mode and a dozen errors say so.
//!
//! 5. **The 16-bit shims are the wrong ones.** `shim/cuda_bf16.h` and
//!    `cuda_fp16.h` carry no `__NV_TL_BUILTIN__` at all; the 13.3 runtime's
//!    carry 120 and 119. With the shims in place the compile reaches tile
//!    codegen and aborts with `Internal Compiler Error (tile codegen):
//!    "Unexpected element type in tile!"` — which is the failure
//!    [`jit::nvrtc`](crate::jit::nvrtc)'s `tile_header_mismatch` exists to
//!    explain, firing here exactly as it was written to. Substituting the real
//!    13.3 headers is what was measured to clear it, and marking the shims
//!    instead is not an option `moe/moe_grouped_gemm_tile.cuh` leaves open:
//!    the shim aliases `__nv_bfloat16` to this tree's own `pie::bf16` so
//!    that FlashInfer stays byte-identical to upstream, and `cuda::tiles`
//!    refuses that type as a tile element whatever it is marked with.
//!
//! 6. **And then the image is Tile IR, not SASS.** With all five answered,
//!    every one of the eleven kernels compiles — and `compile_text` refuses
//!    the result, correctly: *"the image carries `.note.nv.tkinfo` and no
//!    `.text`"*. `nvrtcGetCUBIN` is the wrong call for a tile program. It
//!    wants `nvrtcGetTileIR` and then [`assemble_tile_ir`], which is already
//!    in this crate and has no caller on this path, and `tileiras`, which
//!    ships in its own wheel and is inside no CUDA toolkit here.
//!
//! [`assemble_tile_ir`]: crate::jit::nvrtc::assemble_tile_ir
//!
//! **The device text is not what is missing.** All eleven compile clean under
//! 13.3 with the header, the three options and the real 16-bit headers; the
//! only defect the exercise found was in the environment. That is the same
//! answer the spike at `e99ef19a1` reached from outside the crate with its own
//! C harness, reproduced here through `jit::nvrtc` itself, and it is why these
//! are roots without instantiations rather than dead text: what stands between
//! them and a fire is packaging, and five of the six pieces are ours.
//!
//! # What is deliberately not done here
//!
//! **No `.options(&[..])` on any of the five.** The three options above are
//! necessary and they are nowhere near sufficient — stating them on a root
//! that also needs a header set it cannot ask for would read as a recipe, and
//! the first person to trust it would lose the afternoon this cost.
//!
//! **No instantiation, and no `mod inst` to add one to.** A root with a `mod
//! inst` is compiled by `every_instantiation_compiles` under
//! [`Toolchain::ANY`](crate::jit::Toolchain::ANY), deliberately, so that the
//! fixture asks what the toolchain in front of it can lower rather than
//! declining first. Naming an instantiation here would make that fixture red
//! on this box, which is the correct behaviour and not a state to ship.
//!
//! **The floor is stated and it is a floor.** `.since(13, 3)` is the oldest
//! NVRTC measured to work, not the oldest proven to: 13.1 and 13.2 are not on
//! this machine and were not tried. It earns its place on the fire path
//! regardless — if a `mod inst` ever lands beside one of these, `admits`
//! declines by name instead of handing 13.0 a text it will silently drop.

/// `tile_alternatives.cuh` — the assertions, and the seven kernels it includes.
///
/// The only one of the five that holds no kernel. It is a program source all
/// the same: compiling it is how *"do the bounds still match the sweeps"* gets
/// one answer instead of seven, and its `static_assert`s are the whole of its
/// output.
///
/// Its quoted `#include`s reach `mlp/swiglu_tile.cuh`,
/// `moe/{moe_fused,moe_grouped_gemm,topk_softmax}_tile.cuh`,
/// `norm/{rmsnorm,rmsnorm_rasr}_tile.cuh` and `rope/rope_tile.cuh`, which is
/// why those seven need no root of their own.
pub mod tile_alternatives {
    use crate::jit::Root;

    /// `tile_alternatives.cuh` — the root the predicate assertions compile in.
    pub static ROOT: Root = Root::new("tile/alternatives.cuh");
}

/// `sample/argmax_tile.cuh` — the ARGMAX row's measured representative.
///
/// Not an alternative and never proposed as one: it is the kernel that priced
/// the census bucket at **0.24x**, against the 1.28-1.40x that had been
/// inferred for it from `topk_softmax_tile`. It is carried because a declined
/// verdict backed by a kernel that was written and raced is the thing
/// `tile_alternatives.cuh`'s table is built out of, and deleting the evidence
/// would leave the row an opinion again.
///
/// Reached by no `#include` for the same reason: it belongs to no alternative.
pub mod argmax_tile {
    use crate::jit::Root;

    /// `sample/argmax_tile.cuh` — the root, which nothing here compiles.
    pub static ROOT: Root = Root::new("sample/argmax_tile.cuh");
}

/// `layout/gather_rows_tile.cuh` — the COPY row's measured representative.
///
/// A wash, 0.97-1.10x and bit-identical, which is a result and is why it is
/// here: it is half of the pair that settles what decides an elementwise tile
/// win. A permuted access with no arithmetic is a wash; `rope/rope_tile.cuh`
/// is a permuted access WITH arithmetic and lands with ELEM. Neither sentence
/// survives the other kernel being deleted.
pub mod gather_rows_tile {
    use crate::jit::Root;

    /// `layout/gather_rows_tile.cuh` — the root, which nothing here compiles.
    pub static ROOT: Root = Root::new("layout/gather_rows_tile.cuh");
}

/// `quant/dequant_wna16_tile.cuh` — the DECODE row, and the one that was
/// inferred wrong.
///
/// The census's first version filed it under a `BITS` row reading *"both
/// bandwidth-bound"*, flagged as inferred. Measured, it is 3.5x ahead and
/// bit-identical, because INT4 to bf16 is a 4x expansion. It has no
/// `*_tile_preferred` predicate and so no `static_assert` in
/// `tile_alternatives.cuh` and no `#include` from it — a win that has not been
/// written up as an alternative, rather than one that was declined.
pub mod dequant_wna16_tile {
    use crate::jit::Root;

    /// `quant/dequant_wna16_tile.cuh` — the root, which nothing here compiles.
    pub static ROOT: Root = Root::new("quant/dequant_wna16_tile.cuh");
}

/// `quant/wna16_gemv_tile.cuh` — the GEMV row, measured so the table had no
/// inferred half left.
///
/// The other half of the split `BITS` verdict, and it LOSES at 0.46-0.97x. It
/// was written after `dequant_wna16_tile` proved the inferred row wrong, on
/// the principle that a table with one unmeasured row is a table with a hole;
/// it is the only kernel in this set whose measurement argues for doing
/// nothing, which is exactly why it is not deletable.
pub mod wna16_gemv_tile {
    use crate::jit::Root;

    /// `quant/wna16_gemv_tile.cuh` — the root, which nothing here compiles.
    pub static ROOT: Root = Root::new("quant/wna16_gemv_tile.cuh");
}
