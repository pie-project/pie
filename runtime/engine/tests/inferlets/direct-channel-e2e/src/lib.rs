use inferlet::Result;
use inferlet::ptir::prelude::*;

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let ws = WorkingSet::new();
    let max_pages = 1;
    ws.reserve(max_pages)
        .map_err(|error| format!("ws.reserve: {error}"))?;

    let token = Channel::from(vec![1i32]).named("token");
    let embed_indptr = Channel::from(vec![0u32, 1]).named("embed_indptr");
    let positions = Channel::from(vec![0u32]).named("positions");
    let pages = Channel::from(vec![0u32]).named("pages");
    let page_indptr = Channel::from(vec![0u32, 1]).named("page_indptr");
    let w_slot = Channel::from(vec![0u32]).named("w_slot");
    let w_off = Channel::from(vec![0u32]).named("w_off");
    let kv_len = Channel::from(vec![1u32]).named("kv_len");
    let state = Channel::from(vec![41u32]).named("state");
    let increment = Channel::writer([1], dtype::u32).named("late_increment");
    let out = Channel::new([1], dtype::u32).named("out");

    let pass = ForwardPass::new();
    pass.embed(&token, &embed_indptr)?;
    pass.attention(
        &ws,
        ..,
        ..,
        &kv_len,
        &pages,
        &page_indptr,
        &w_slot,
        &w_off,
        &positions,
        None,
    )?;
    pass.epilogue(move || {
        let current = state.take().tensor();
        let late_increment = increment.take().tensor();
        let next = add(&current, &late_increment);
        state.put(&next);
        out.put(&next);
    });

    let pipeline = Pipeline::new();
    pass.submit(&pipeline)
        .map_err(|error| format!("submit: {error}"))?;
    increment.put(vec![1u32]);
    let value = out
        .take()
        .get::<u32>()
        .await
        .map_err(|error| format!("take: {error}"))?[0];
    pipeline.close();
    pass.rs_working_sets(&[])
        .map_err(|error| format!("in-place pure RS rebind: {error}"))?;

    Ok(format!("value={value}"))
}
