/// The `[model] device_kv_budget` that buys `pages` full pages of 16 tokens
/// for `trace`'s pools, `slots` state slots beside them: the few sequences a
/// test seats, stated in bytes as a deployment states them.
pub fn kv_budget(trace: &poem_ir::Trace, context: u32, slots: u32, pages: u32) -> Option<u64> {
    let paging = engine_xla::store::kv::Paging::of(16, context, slots, u64::from(pages))
        .expect("pages of 16 tokens");
    Some(engine_xla::store::pool_demand(trace, paging).expect("the pools size"))
}
