//! `adapter` — the LoRA seam, which has no device text at all.
//!
//! One symbol, no `.cuh`, no `__global__`, no `unit!` and **no `bind!`**.
//! §5 step 5's third family, and the one that exists to prove the floor can
//! describe a kernel it does not own.
//!
//! # What this replaces
//!
//! ```text
//!   before                                              lines
//!   kernels-cuda-new/src/families/adapter.rs  0 rows        72
//!   kernels-cuda-new/src/table/adapter.rs     1 row         57
//!                                                       ------
//!                                                         129
//!   after
//!   kernels-cuda-new/src/x/adapter.rs         1 contract
//! ```
//!
//! There is no `fire/adapter.rs` to move, because there is nothing here to
//! fire. That is the whole subject of this file.
//!
//! # A symbol with no kernel
//!
//! `pie_lora_qkv_correction` is a **driver op**: `bind/mod.rs` matches it by
//! hand and runs a batched-GEMM sequence — `x @ A^T` then `@ B^T` — through
//! cuBLAS, scaled and accumulated into an existing qkv projection. The
//! `pie_` prefix, which every other symbol in the tree earns by being a
//! `pie_k_` C shim, here means the opposite: **there is no shim because
//! there is no kernel**.
//!
//! `families/adapter.rs`'s header said it in one line and it survives:
//!
//! > The correction is a sequence of cuBLAS batched GEMMs, not a
//! > `__global__`, so there is nothing for NVRTC to compile and nothing for a
//! > `unit!` to name.
//!
//! # And yet it is declared — because of who reads a declaration
//!
//! §1's ladder is a ladder of READERS, not of implementations. The reader
//! this contract serves is `model-compiler`, which is GPU-free and **must
//! not be able to tell a cuBLAS symbol from a JIT'd one**. `check_plan`
//! refuses a symbol nothing declares; if `pie_lora_qkv_correction` were
//! absent from `SIGS`, every LoRA trace in the tree would be refused at
//! compile time by a compiler whose whole job is to not know how a symbol is
//! implemented.
//!
//! `x/mod.rs`'s note on `SIGS` is the mechanism: **`FAMILIES` is `_cuda`-gated
//! and `SIGS` is not**, so a contract can reach `model-compiler` on a machine
//! with no GPU and no CUDA toolkit, which is where `model-compiler` is
//! usually built.
//!
//! # The third registration shape, worked
//!
//! `x/mod.rs`'s "three registration shapes" names this file as the driver-op
//! example. Restated from the other side:
//!
//! | this file | why |
//! | --- | --- |
//! | `UNITS = &[]` | no device text to compile |
//! | `contract!`, so `SIGS` has it | `model-compiler` must not know the difference |
//! | **not** in `x::FAMILIES` | see below — `route()` must keep answering `Driver` |
//! | no `bind!`, so no `ENTRIES` | a bind launches a kernel; this is not one |
//!
//! # Why NOT `FAMILIES`, precisely — and it is `route()`'s one overlap
//!
//! An `Entry` would make [`crate::x::route`] answer `Route::Bound` for this
//! symbol, and a bound symbol is fired by `x::fire` from a `Cx`. A `Cx` is a
//! read-only query facade over a fire: it can answer `arg_in(0)`, `rows()`
//! and `weight_named("lora_a")`, and it **cannot hand out a cuBLAS handle**,
//! because a handle is a mutable, per-device, per-stream resource that
//! `driver-cuda` owns and `kernels-cuda-new` has never seen. So a bind body
//! could read every operand this op needs and still not be able to run it.
//!
//! Leaving `ENTRIES` empty is therefore not an omission to be tidied later.
//! It is the mechanism by which `route()` keeps answering `Route::Driver`
//! for this symbol — `x/mod.rs`'s "THE ONE OVERLAP" — and the hand-written
//! match in `bind/mod.rs` keeps firing it. **A future editor who adds this
//! family to `FAMILIES` for symmetry breaks LoRA on every model in the
//! tree**, and the failure is a refusal at model load rather than a wrong
//! answer, which is the only comfort available.
//!
//! What would close it: a `Cx` that can lend a BLAS handle, at which point
//! the seam becomes a two-call `fn` body like any other §2.3 composition.
//! That is a floor change with one caller, which is below §10.5's bar until
//! a second driver op wants it.
//!
//! # The operands, and the two that a row could not source
//!
//! The deleted row stated eight operands. Six had sources. The two that did
//! not are the shape of every `none:` in this port, and the row's own words
//! are kept:
//!
//! > `lora_a` and `lora_b` are per-adapter weights selected at RUN TIME by
//! > the request's adapter id, not weights the statement names. No `Source`
//! > names "the adapter this request asked for", because the trace does not
//! > know it — the same trace serves every adapter.
//!
//! This is not a `none:` arm here only because there is no `bind!` to put an
//! arm in: the refusal `Route::Driver` makes is not "cannot", it is "not
//! mine", and `bind/mod.rs`'s hand-written match resolves the adapter id
//! from the request rather than from the trace. The row's operand list was
//! describing a binding that was never going to happen, which is the clearest
//! single argument in this family for why operands leave the declaration.
//!
//! # The one live caller
//!
//! `dsl::cuda::lora_qkv_correction` is called from `dsl.rs`'s `pub fn seam`,
//! which is reached by `gemma_4` and by `llama_like`. This symbol is fired
//! in production and nothing about it changes here — only where its
//! declaration lives.

use crate::unit::Unit;

/// No device text.
///
/// Stated rather than omitted, because `families/mod.rs::ALL` concatenates
/// `UNITS` from every family and an absent name is a compile error where an
/// empty slice is a fact. It is also the answer to *"which units does
/// `adapter` compile"* for anyone who greps, and the answer is none.
pub static UNITS: &[Unit] = &[];

// ---------------------------------------------------------------------------
// The declaration, and the whole of this family.
// ---------------------------------------------------------------------------

contract! {
    /// Applies a LoRA correction to a fused qkv projection, in place.
    ///
    /// `qkv += ((x @ lora_a^T) @ lora_b^T) * scale`, with `lora_a`/`lora_b`
    /// selected per request by adapter id and the whole sequence run as
    /// batched cuBLAS GEMMs. `pie_`-prefixed because it is a driver op and
    /// not a `pie_k_` C shim — there is no kernel to shim.
    ///
    /// Fired by `bind/mod.rs`'s hand-written match, reached through
    /// [`route`](crate::x::route) answering `Route::Driver`. See this
    /// module's header for why an `Entry` here would break that.
    LORA_QKV_CORRECTION = "pie_lora_qkv_correction" as lora_qkv_correction
}
