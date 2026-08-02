use inferlet::Result;
use inferlet::ptir::attention::prelude::*;

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let ws = WorkingSet::new();
    let max_pages = 1;
    ws.reserve(max_pages).context("ws.reserve")?;

    let token = Channel::from([1i32]).named("token");
    let embed_indptr = Channel::from([0u32, 1]).named("embed_indptr");
    let positions = Channel::from([0u32]).named("positions");
    let pages = Channel::from([0u32]).named("pages");
    let page_indptr = Channel::from([0u32, 1]).named("page_indptr");
    let w_slot = Channel::from([0u32]).named("w_slot");
    let w_off = Channel::from([0u32]).named("w_off");
    let kv_len = Channel::from([1u32]).named("kv_len");
    let state = Channel::from([41u32]).named("state");
    let increment = Channel::writer([1], dtype::u32).named("late_increment");
    let out = Channel::new([1], dtype::u32).named("out");

    let pass = ForwardPass::new();
    pass.embed(&token, &embed_indptr)?;
    pass.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &kv_len,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &positions,
            mask: None,
        },
    )?;
    pass.epilogue(move || {
        let current = state.take();
        let late_increment = increment.take();
        let next = &current + &late_increment;
        state.put(&next);
        out.put(&next);
    });

    let pipeline = Pipeline::new();
    // The late-put run-ahead pattern (submit first, value later) is k = 1
    // semantics; frame deployments require every per-fire input staged
    // before submit (Vesuvius §5), so pre-stage there.
    let late_put = frame_size() == 1;
    if !late_put {
        increment.put(vec![1u32]);
    }
    pass.submit(&pipeline).context("submit")?;
    if late_put {
        increment.put(vec![1u32]);
    }
    let value = out.take_host::<u32>().await?;
    pipeline.close();

    Ok(format!("value={value}"))
}
