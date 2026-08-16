//! The CuTile roots: five program sources, no [`Family`](crate::jit::Family),
//! no routine, no `mod inst`. A [`Root`](crate::jit::Root) is this crate's only
//! way of saying a file is a program source; without these, twelve carried
//! `.cuh` read as orphans in the gap between carried and compiled-against.
//!
//! Six measured blockers stand between them and a cubin (L40S, 2026-08-16):
//! NVRTC loads as 13.0.88; `crt/cuda_tile.h` resolves under neither toolkit,
//! since this crate passes no `-I` by design; `-std=c++17` is hard-coded and
//! that header `#error`s below C++20; `-enable-tile`'s absence is a WARNING, so
//! the compile succeeds having produced nothing; `shim/cuda_bf16.h` carries no
//! `__NV_TL_BUILTIN__` (the ICE `jit::nvrtc`'s `tile_header_mismatch` explains)
//! and cannot be marked away, since it aliases `__nv_bfloat16` to `pie::bf16` to
//! keep FlashInfer identical to upstream; and the image is Tile IR, wanting
//! `tileiras`. Hence no `.options`, no `mod inst` (`every_instantiation_compiles`
//! would go red here), and a `.since(13, 3)` floor measured, not proven.

/// Assertions only: pins each `*_tile_preferred` bound to its sweep row, and reaches the other seven `.cuh` by quoted `#include`, so they need no root.
pub mod tile_alternatives {
    use crate::jit::Root;

    pub static ROOT: Root = Root::new("tile/alternatives.cuh");
}

/// The ARGMAX row: **0.24x** measured, against 1.28-1.40x inferred from `topk_softmax_tile`.
pub mod argmax_tile {
    use crate::jit::Root;

    pub static ROOT: Root = Root::new("sample/argmax_tile.cuh");
}

/// The COPY row: a wash, 0.97-1.10x; paired with `rope/rope_tile.cuh` it shows arithmetic, not permutation, decides an ELEM win.
pub mod gather_rows_tile {
    use crate::jit::Root;

    pub static ROOT: Root = Root::new("layout/gather_rows_tile.cuh");
}

/// The DECODE row: 3.5x ahead, because INT4 to bf16 is a 4x expansion; no `*_tile_preferred` predicate, so a win not yet written up.
pub mod dequant_wna16_tile {
    use crate::jit::Root;

    pub static ROOT: Root = Root::new("quant/dequant_wna16_tile.cuh");
}

/// The GEMV row, the other half of the split `BITS` verdict: it LOSES at 0.46-0.97x.
pub mod wna16_gemv_tile {
    use crate::jit::Root;

    pub static ROOT: Root = Root::new("quant/wna16_gemv_tile.cuh");
}
