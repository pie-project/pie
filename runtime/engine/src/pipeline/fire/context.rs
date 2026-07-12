//! The narrow surface [`fire`](super)'s orchestration needs from its caller:
//! WASM component resource-table access (to get/get_mut/delete/push
//! `Resource<Channel>`/`Resource<ForwardPass>`/`Resource<Pipeline>` handles)
//! and the calling process's identity (the reclaim ladder's FCFS key). This
//! trait names no `inferlet`/`ProcessCtx` type — only the external
//! `wasmtime::component::ResourceTable` leaf type and `uuid::Uuid` — so
//! `pipeline/` stays strictly below `inferlet/` in the layering.
//! `inferlet::host` (L4) implements it for `ProcessCtx`.

use wasmtime::component::ResourceTable;

/// The fire engine's view of its caller: a WASM component resource table
/// plus the caller's process identity. Implemented for `ProcessCtx` in
/// `inferlet::host`; every `pipeline::fire` orchestration function is generic
/// over `C: FireContext` instead of naming `ProcessCtx` directly.
pub trait FireContext {
    /// The WASM component resource table (owns the `Resource<Channel>` /
    /// `Resource<ForwardPass>` / `Resource<Pipeline>` storage this module
    /// operates on).
    fn resources(&mut self) -> &mut ResourceTable;

    /// This process's identity — the reclaim ladder's FCFS key
    /// (`crate::store::reclaim::ProcessId` / `crate::scheduler::ProcessId`
    /// are the same `uuid::Uuid` representation; returned as the leaf-crate
    /// type directly so this trait need not name either module).
    fn process_id(&self) -> uuid::Uuid;

    /// Process-owned KV membership captured independently of WIT handles.
    fn kv_working_sets(
        &self,
    ) -> std::collections::HashSet<crate::store::kv::page_table::WorkingSetId>;

    /// Honor a requester self-suspend decision while this task still owns the
    /// process continuation.
    async fn honor_preemption(&mut self) -> anyhow::Result<()>;
}
