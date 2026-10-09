// `runtime::model::register` reads a model's depth, vocabulary and arch from
// its package, so every model a deployment is listed for states them.
#[test]
fn every_listed_model_states_its_depth_and_arch() {
    let missing: Vec<&str> = models::deployments()
        .map(|d| d.entry)
        .filter(|entry| entry.layers == 0 || entry.arch.is_empty())
        .map(|entry| entry.id)
        .collect();
    assert!(missing.is_empty(), "no depth or arch for {missing:?}");
}
