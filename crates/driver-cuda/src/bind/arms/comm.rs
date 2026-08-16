//! What a trace that states a tensor-parallel collective binds to.
//!
//! `comm::` is the custom P2P reduction over IPC-mapped peer memory and
//! `dist::` is NCCL; a sharded model text guards between them by message size.
//! No row here carries a hand arm, and the two families lost theirs for
//! opposite reasons.
//!
//! `dist::` CROSSED. NCCL resolves its own communicator inside the collective,
//! so the arms only ever computed `rows x width` -- arithmetic a launcher owns
//! (F6). Their `ROUTINES` rows are `routine!` now and the column binds them.
//!
//! `comm::` went the other way, to [`Bound::driver`]. The custom reduction
//! needs the `car` handle itself and not a resolved plane: `plane_for` reads
//! the launch's input and the stream's CAPTURE STATE to pick a `RankData`
//! slot, and `note_graph_buffer` needs `&mut`. A query-only `Cx` may offer
//! neither, which is what the driver-op table in `bind/mod.rs` is for.
//!
//! A decline is a ROUTING answer, not a failure. `comm::all_reduce_bf16` falls
//! back to NCCL inside the launcher; the fused
//! `all_reduce_residual_rmsnorm_bf16` refuses instead, there being no fused
//! NCCL symbol and the unfused composition landing its sum in the PARTIAL
//! rather than the residual operand this statement aliases.
//!
//! # Nothing here registers a buffer
//!
//! Registering would DEADLOCK: `register_buffer` returns early when the base
//! is known and runs an all-gather when it is not, so a rank that skips the
//! gather leaves its peers blocked forever -- and ranks need not first-sight
//! the same address at the same statement. It belongs in a setup step every
//! rank runs unconditionally.
//!
//! A collective needs a peer and this machine has one GPU: the refusal mapping
//! is type-checked against every `Decline`, but no fire has run these.

use super::Bound;


/// The tensor-parallel collectives.
///
/// The `dist::` three are derived: their column IS their signature, and
/// `kernels_cuda::dist` still refuses at the fire for want of an NCCL
/// binding -- a refusal about the BUILD, not about the binding.
pub static ARMS: &[Bound] = &[
    Bound::driver("comm::all_reduce_bf16"),
    Bound::driver("comm::all_reduce_residual_rmsnorm_bf16"),
    Bound::derived("dist::all_reduce_bf16"),
    Bound::derived("dist::all_reduce_bf16_out"),
    Bound::derived("dist::all_gather_bf16"),
];

