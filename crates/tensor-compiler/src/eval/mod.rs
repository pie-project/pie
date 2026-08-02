//! # `tensor-compiler` — the PTIR semantics oracle
//!
//! What a program *produces*, computed on the host. Two consumers, one set of op
//! semantics:
//!
//! * [`interp`] — the **tier-0 reference interpreter**: the golden model every
//!   backend diffs against. It pins the rules that no backend may reinterpret —
//!   the readiness rule, register-semantics commit (take pops once, last put
//!   wins), `argmax` tie-break, `sort_desc` NaN order, left-aligned broadcast,
//!   and the `splitmix64` / `hash_uniform` RNG.
//! * [`pareval`] — **host partial evaluation** of stage programs, reusing the
//!   interpreter's `eval_op` so there is no second evaluator and no drift.
//!
//! Despite the name, this is not a test-only crate. `pareval` is a production
//! path with three callers: canonical-KV fire evidence (the prefix cache folds
//! the geometry prologue instead of pattern-matching the trace), capability-less
//! execution (a driver with no device-geometry ports has the host fold the
//! prologue per fire), and geometry classification
//! ([`pareval::geometry_taint`], which decides whether a pass's submission
//! geometry is host-knowable by derivability rather than op-pattern arity).

// The modules were authored against `alloc` paths in the `no_std` IR crate and
// still use them; `alloc` is available here through `std`.

pub mod interp;
pub mod pareval;
