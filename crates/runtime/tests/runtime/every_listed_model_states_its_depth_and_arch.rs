// `runtime::model::register` reads a model's depth, vocabulary and arch from
// its package, so every model a deployment is listed for states them.
#[test]
fn every_listed_model_states_its_depth_and_arch() {
    let missing: Vec<&str> = runtime::catalog::deployments()
        .map(|d| &d.model)
        .filter(|model| model.layers == 0 || model.arch.is_empty())
        .map(|model| model.id.as_str())
        .collect();
    assert!(missing.is_empty(), "no depth or arch for {missing:?}");
}
