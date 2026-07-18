//! Real component-boundary coverage for explicit context destruction.

use futures::future::join_all;
use inferlet::{Context, Result, model::Model, runtime};

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let models = runtime::models();
    let model = Model::load(&models[0])?;
    let root = Context::new(&model)?;
    let forks = (0..4).map(|_| root.fork()).collect::<Result<Vec<_>>>()?;

    join_all(forks.into_iter().map(|fork| async move {
        fork.destroy();
    }))
    .await;
    root.destroy();

    if input == "error" {
        return Err("injected error after context cleanup".into());
    }
    Ok("contexts released".into())
}
