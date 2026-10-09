use poem::{Operands, Platform, Trace, ValueId};

const ROWS: [&str; 3] = [
    "wan22-ti2v-5b-bf16-kv-bf16",
    "wan22-mini-d128-bf16-kv-bf16",
    "wan22-mini-nano-bf16-kv-bf16",
];

const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

fn trace(deployment: &str, platform: Platform) -> Trace {
    let row = models::deployment(deployment)
        .unwrap_or_else(|| panic!("this build ships no `{deployment}`"));
    row.trace(platform)
}

fn last_read(t: &Trace) -> Vec<Option<usize>> {
    let mut last = vec![None; t.values.len()];
    let mut ins = Vec::new();
    for (j, node) in t.nodes.iter().enumerate() {
        ins.clear();
        node.op.inputs(&mut ins);
        for &ValueId(id) in &ins {
            last[id as usize] = Some(j);
        }
    }
    last
}

#[test]
fn every_wan_2_fold_owns_the_vector_it_folds_into() {
    for deployment in ROWS {
        for platform in PLATFORMS {
            let t = trace(deployment, platform);
            let last = last_read(&t);
            let mut pairs = Vec::new();
            let mut folds = 0usize;
            for (j, node) in t.nodes.iter().enumerate() {
                pairs.clear();
                node.op.aliases(&mut pairs);
                for &(_out, ValueId(input)) in &pairs {
                    folds += 1;
                    assert!(
                        last[input as usize] <= Some(j),
                        "{deployment}/{platform:?}: node {j} `{}` folds in place into v{input}, \
                         which node {:?} reads afterwards — an in-place fold must own \
                         its operand (add to a copy, as `forward::copy_of` does)",
                        node.op.name(),
                        last[input as usize],
                    );
                }
            }
            assert!(
                folds > 0,
                "{deployment}/{platform:?}: this plan folds nothing in place, so the claim \
                 is vacuous and the walk above must be wrong"
            );
        }
    }
}
