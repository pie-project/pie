use crate::inferlet::ProcessCtx;
use crate::inferlet::host::pie;

use anyhow::Result;

impl pie::inferlet::system::Host for ProcessCtx {
    async fn version(&mut self) -> Result<String> {
        Ok(env!("CARGO_PKG_VERSION").to_string())
    }

    async fn instance_id(&mut self) -> Result<String> {
        Ok(self.id().to_string())
    }

    async fn username(&mut self) -> Result<String> {
        Ok(self.get_username())
    }

    async fn declare_restartable(&mut self) -> Result<()> {
        // Run again from its input, it would find its handles spent.
        if self.handles().is_none() {
            crate::inferlet::process::declare_restartable(self.id());
        }
        Ok(())
    }
}
