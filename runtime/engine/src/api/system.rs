//! pie:inferlet/system - runtime information (version, instance-id, username).
//!
//! Timing moved to the standard `wasi:clocks/monotonic-clock@0.3` import in
//! Phase 3 (native async `wait-for`), so the host timer that backed the old
//! `system.sleep` is gone.

use crate::api::pie;
use crate::inferlet::ProcessCtx;

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
}
