//! What the load's own bind walk builds (a rebased twin, under its kernel's
//! name) is the load's, never counted as a fire's; what is built outside the
//! walk, before it or after, is counted, once.

use engine_gpu::serve::Serve as _;

use crate::backend::{Api, Shell};
use crate::model_logits::{CONTEXT, fixture, load};

/// Builds baseline kernel `name`, which no fire of the load asks for, as a
/// fire that reached one no walk named would.
fn build(shell: &Shell, name: &str) -> Vec<&'static str> {
    let key = kernels_slang::key(name, kernels_slang::Tier::Baseline).expect("instantiated");
    shell
        .pipelines()
        .get(shell.device(), "elemwise/norm.slang", key)
        .expect("the kernel builds");
    vec![key]
}

#[test]
fn the_bind_walks_builds_are_the_loads_and_any_other_is_counted() {
    let Some(mut shell) = load(&fixture("qwen3-micro-text", None), CONTEXT) else {
        return;
    };
    shell.built_serving();
    let early = build(&shell, "elemwise_norm_layernorm_bf16");
    let before = Api::compiled(shell.pipelines());
    shell.walk_binds().expect("the bind walk");
    let walked = Api::compiled(shell.pipelines()) - before;
    eprintln!("the bind walk built {walked}");
    assert!(walked > 0, "the walk builds a rebased twin of its own");
    assert_eq!(
        shell.built_serving(),
        early,
        "the walk's builds are the load's, and one made before it stays owed"
    );
    let late = build(&shell, "elemwise_norm_layernorm_unscaled_bf16");
    assert_eq!(shell.built_serving(), late, "a build after the walk is a fire's");
    assert!(shell.built_serving().is_empty(), "each is counted once");
}
