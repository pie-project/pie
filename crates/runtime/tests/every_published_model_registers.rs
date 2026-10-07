// `runtime::model::register` reads a model's depth, vocabulary and arch from
// ROWS by its catalog id, whatever deployment of it the engine loaded; a model
// the table does not carry boots and refuses at registration.
#[test]
fn every_published_model_registers() {
    let missing: Vec<&str> = models::entries()
        .filter(|entry| !entry.mini)
        .map(|entry| entry.id)
        .filter(|id| runtime::model::row(id).is_none())
        .collect();
    assert!(
        missing.is_empty(),
        "`runtime::model::ROWS` states no row for {missing:?}"
    );
}
