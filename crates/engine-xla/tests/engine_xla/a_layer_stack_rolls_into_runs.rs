//! A trace's layers fall into runs of a repeating body (`roll::runs`), the
//! stack a fire program can emit once as a loop instead of once a layer:
//! a dense stack is one body (its first layer included: the accumulator
//! ScoreCapture starts there is the one later layers carry); a stack of
//! sliding and global attention repeats a period of six, a hybrid of
//! recurrent and attention layers a period of four. Needs no device: the
//! catalog's traces.

use engine_xla::roll::{self, Run};
use poem::Platform;

fn runs_of(name: &str) -> (u32, Vec<Run>) {
    let deployment = poem_compiler::catalog::deployments()
        .find(|d| d.name == name)
        .unwrap_or_else(|| panic!("the catalog lists `{name}`"));
    let trace = deployment.trace(Platform::Xla);
    (roll::layer_count(&trace), roll::runs(&trace))
}

#[test]
fn a_layer_stack_rolls_into_runs() {
    let one = |name: &str, layers: u32, period: u32| {
        let (count, runs) = runs_of(name);
        assert_eq!(count, layers, "{name}");
        assert_eq!(
            runs,
            vec![Run {
                first: 0,
                period,
                repeats: layers / period
            }],
            "{name} rolls into one body of {period} layer(s)"
        );
    };
    one("qwen3-0.6b-bf16-kv-bf16", 28, 1);
    one("llama31-8b-bf16-kv-bf16", 32, 1);
    one("qwen3-30b-a3b-bf16-kv-bf16", 48, 1);
    one("gemma4-31b-bf16-kv-bf16", 60, 6);
    one("qwen35-d0.8b-bf16-kv-bf16", 24, 4);
    // every layer is in exactly one run, in order
    for (name, (layers, runs)) in [
        (
            "qwen3-0.6b-bf16-kv-bf16",
            runs_of("qwen3-0.6b-bf16-kv-bf16"),
        ),
        ("glm45-air-bf16-kv-bf16", runs_of("glm45-air-bf16-kv-bf16")),
    ] {
        let mut next = 0;
        for r in &runs {
            assert_eq!(r.first, next, "{name}: runs tile the stack: {runs:?}");
            next += r.layers();
        }
        assert_eq!(next, layers, "{name}: runs cover every layer");
    }
}
