//! The narrow surface [`fire`](super)'s orchestration needs from its caller:
//! WASM resource-table access and the caller's process identity. Naming no
//! `inferlet` type keeps `pipeline/` strictly below `inferlet/`.

use wasmtime::component::ResourceTable;

/// The fire engine's view of its caller; orchestration is generic over this.
pub trait FireContext {
    /// The table owning the `Resource<Channel>` / `<ForwardPass>` / `<Pipeline>`
    fn resources(&mut self) -> &mut ResourceTable;

    /// This process's identity — the planner's FCFS key.
    fn process_id(&self) -> uuid::Uuid;

    /// Settle this process's own tail: only the owning guest task can, since
    /// finalization needs its ResourceTable.
    async fn settle_pipeline_tail(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}
