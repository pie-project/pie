use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

fn bx<T>(value: T) -> &'static T {
    Box::leak(Box::new(value))
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    model::configure(wit_model::output_vocab_size(), 16, 1);

    let ws: &'static WorkingSet = bx(WorkingSet::new());
    ws.reserve(1)
        .map_err(|error| format!("ws.reserve: {error}"))?;

    let token = bx(Channel::from(vec![1i32]).named("token"));
    let klen = bx(Channel::from(vec![1u32]).named("klen"));
    let state = bx(Channel::from(vec![41u32]).named("state"));
    let increment = bx(Channel::writer([1], dtype::u32).named("late_increment"));
    let out = bx(Channel::new([1], dtype::u32).named("out"));

    let pass: &'static ForwardPass<'static> = bx(ForwardPass::new());
    pass.embed(token, Tensor::constant(vec![0u32, 1]));
    pass.attn_working_set(ws, klen);
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
    pass.set_rs_working_sets(&[])
        .map_err(|error| format!("in-place pure RS rebind: {error}"))?;

    Ok(format!("value={value}"))
}
