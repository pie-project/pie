//! A package states its template as a format and settings; a format the
//! runtime does not ship, or a setting the format does not read, is a fault
//! the catalog would only find at serve time, so the repository's are
//! checked here.

#[test]
fn every_model_is_spoken_through_a_format_the_runtime_ships() {
    let mut faults = Vec::new();
    for (package, model) in poem_compiler::catalog::repository().models() {
        let t = &model.template;
        let spec = chat_template::Spec {
            format: t.format.clone(),
            thinking: t.thinking,
            preserve_thinking: t.preserve_thinking,
            tools: t.tools,
            generation_suffix: t.generation_suffix.clone(),
            stop: t.stop.clone(),
            bos: t.bos.clone(),
            eos: t.eos.clone(),
        };
        if let Err(why) = spec.check() {
            faults.push(format!("{}/{}: {why}", package.name(), model.id));
        }
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}
