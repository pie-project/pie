//! Reading op-tag dispatch back out of the hand-written runtime sources.
//!
//! The `.cuh` and `.metal` files under `runtime/` are the other half of the
//! emitter: Rust decides which helper an op is routed to, and C++ decides
//! what that helper does with it. Nothing in either language checks that
//! the two agree, and the failure is quiet, so the runtime sources are
//! parsed here and compared against [`eta_ir::op::OP_TABLE`].


