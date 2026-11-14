use wasmtime::component::ResourceTable;

pub trait FireContext {
    fn resources(&mut self) -> &mut ResourceTable;

    fn process_id(&self) -> uuid::Uuid;

    /// A fire of this process's found `rows` already cached and mapped them
    /// in rather than computed them.
    fn note_cached(&mut self, _rows: u64) {}

    async fn settle_pipeline_tail(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// The pending fires of every pipeline whose residency this process
    /// holds: the tail a windowed ask finalizes as it lands.
    fn pipeline_tails(&self) -> Vec<crate::pipeline::fire::PendingFires> {
        Vec::new()
    }
}
