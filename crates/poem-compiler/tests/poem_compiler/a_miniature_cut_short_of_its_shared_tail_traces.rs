//! A gemma miniature cut to fewer layers than its shared tail begins at
//! (a checkpoint stating a shallower depth, as a catalog walk over another
//! model's snapshot cuts every miniature) traces on the backends that read
//! a readout row as its token: the fold never reaches the tail, so its rows
//! are read out unsplit.

use poem::Platform;

#[test]
fn a_miniature_cut_short_of_its_shared_tail_traces() {
    for (name, layers) in [("gemma4-e4b-mini-l30-bf16-kv-bf16", 24), ("gemma4-e2b-bf16-kv-bf16", 6)] {
        let mut cut = poem_compiler::catalog::deployment(name).expect("listed").clone();
        cut.deploy.layers = Some(layers);
        for platform in [Platform::Metal, Platform::Xla] {
            cut.package
                .trace(&cut.model.id, &cut.deploy, &cut.name, platform)
                .unwrap_or_else(|why| panic!("{name} at {layers} layers on {platform:?}: {why:#}"));
        }
    }
}
