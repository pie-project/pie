# `driver-metal` — retired, kept for reference

**This crate is no longer built, tested, or maintained.** It was removed from
the workspace members and from `engine`'s dependency graph on 2026-08-10. The
source stays because it is the record of what the Metal driver did and why —
several of its comments are the only account of bugs that were paid for once.

## Why

Two decisions, in order.

1. **The Rust rewrite.** `crates/driver-metal-new` is the replacement. Its
   `PARITY*.md` ledgers are the entity-by-entity account of what came across,
   what was deliberately dropped, and what a Rust type made unrepresentable.
   Read those before this.
2. **The model-compiler path.** `crates/driver-metal-new/DIRECTION.md`: a
   traced fire is lowered to a flat list of launches, each naming a kernel
   symbol, and the driver binds and calls. Nothing in a driver chooses a
   kernel. That retires the *shape* of this crate, not just its language — the
   per-family forwards under `csrc/src/model/` and `csrc/src/batch/` have no
   successor to be ported into.

## What this means for a reader

* Nothing here is on any serving path. There is no `pie_metal_*` symbol in the
  engine any more; `engine/src/driver/backend/metal.rs` is deleted.
* It does not build. Its `csrc/` needs CMake and an Apple toolchain, and no CI
  job invokes either.
* **Do not port from it by default.** The ledgers say what was already taken.
  The remaining C++ is mostly the family executor (`batch/forward.cpp`), which
  the lowering replaces rather than translates.

## Still worth reading

The comments that record measurements and failures, because they are the only
copy:

* `csrc/src/batch/forward.cpp` — the unkillable-process story behind
  `fits_on_this_gpu`; the recurrent-slot budget that reserved 10.6 GiB because
  a ceiling was read as a floor; the dense-mask materialisation that cost
  8.4 MB per step for a buffer no kernel read.
* `csrc/src/pipeline/interp.hpp` — states four times that it is a hand copy of
  `interp.rs`, which is how the three-copies-of-one-golden-model problem was
  found.
* `csrc/src/batch/expert_paging.hpp` — why an Apple GPU with no demand paging
  turns one step into several command buffers.
