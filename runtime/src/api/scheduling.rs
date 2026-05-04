//! pie:core/scheduling - Market prices and program account queries

use crate::api::pie;
use crate::api::model::Model;
use crate::api::context::Context;
use crate::context;
use crate::instance::InstanceState;

use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl pie::core::scheduling::Host for InstanceState {
    // ── Market ──────────────────────────────────────────────────────

    /// Make cost: constant price to produce one new KV page.
    async fn price(&mut self) -> Result<f64> {
        Ok(1.0)
    }

    /// Rent (clearing price) on this context's device, last step.
    async fn rent(&mut self, ctx: Resource<Context>) -> Result<f64> {
        let ctx = self.ctx().table.get(&ctx)?;
        let model_idx = ctx.model_id;
        let context_id = ctx.context_id;

        let device = context::get_device(model_idx, context_id);
        Ok(context::get_clearing_price(model_idx, device))
    }

    /// Dividend this process received last step.
    /// = dividend_per_endowment × process_endowment  (§3.4)
    async fn dividend(&mut self, model: Resource<Model>) -> Result<f64> {
        let model_idx = self.ctx().table.get(&model)?.model_id;
        let pid = self.id();
        let dividend_rate = context::get_dividend_rate(model_idx);
        let endowment = context::get_endowment(model_idx, pid);
        Ok(dividend_rate * endowment)
    }

    // ── Device ──────────────────────────────────────────────────────

    /// Per-tick latency of this context's device (seconds), updated each tick.
    async fn latency(&mut self, ctx: Resource<Context>) -> Result<f64> {
        let ctx = self.ctx().table.get(&ctx)?;
        let model_idx = ctx.model_id;
        let context_id = ctx.context_id;

        let device = context::get_device(model_idx, context_id);
        Ok(context::get_tick_latency(model_idx, device))
    }

    // ── Account ────────────────────────────────────────────────────

    async fn balance(&mut self, model: Resource<Model>) -> Result<f64> {
        let model_idx = self.ctx().table.get(&model)?.model_id;
        let pid = self.id();
        Ok(context::get_balance(model_idx, pid))
    }
}
