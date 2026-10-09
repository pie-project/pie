use poem::Platform;
use poem_ir::{Attention, Operation, Trace};

fn carries_a_head(deployment: &str) -> bool {
    models::published::all().any(|p| p.deployment == deployment)
}

fn masked_arms(trace: &Trace) -> usize {
    trace
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.op,
                Operation::Attention(Attention::Masked { .. } | Attention::MaskedLse { .. })
            )
        })
        .count()
}

#[test]
fn the_masked_axis_is_declared_by_gemma_and_qwen_and_by_nobody_else() {
    const DECLARE: [&str; 8] = [
        "gemma4-",
        "diffusiongemma-",
        "gptoss-",
        "hunyuanimage3-",
        "muse-glimmer-",
        "qwen35-",
        "qwen36-",
        "qwen38-",
    ];
    const GAPPED: [&str; 3] = ["dsv4-", "glm5-", "kimik3-"];
    const MASKLESS_RIG: &str = "glm5-a12b-bf16-kv-bf16";

    let mut declaring: Vec<(String, usize)> = Vec::new();
    let mut maskless: Vec<String> = Vec::new();
    for row in models::deployments().chain(models::splits()) {
        let deployment = row.name.as_str();
        let arms = masked_arms(&row.trace(Platform::Cuda));
        if arms > 0 {
            declaring.push((deployment.to_string(), arms));
        } else {
            maskless.push(deployment.to_string());
        }
    }

    assert!(
        declaring.iter().all(|(deployment, _)| {
            DECLARE.iter().any(|family| deployment.starts_with(family))
                || carries_a_head(deployment)
        }),
        "a family beyond gemma and qwen declares `attention.masked` from its \
         own text — not from an overlaid drafter head — and the device gates \
         in this file were written against gemma: {declaring:?}"
    );
    assert!(
        !declaring.is_empty(),
        "no deployment declares `attention.masked` at all, and then the axis has no \
         model text to be exercised by"
    );

    for family in DECLARE {
        assert!(
            declaring
                .iter()
                .any(|(deployment, _)| deployment.starts_with(family)),
            "no `{family}*` deployment declares `attention.masked` any more, so the \
             axis lost a family: {declaring:?}"
        );
    }

    for family in GAPPED {
        let grew: Vec<&(String, usize)> = declaring
            .iter()
            .filter(|(deployment, _)| deployment.starts_with(family) && !carries_a_head(deployment))
            .collect();
        assert!(
            grew.is_empty(),
            "`{family}*` grew an `attention.masked` arm in its own text, and a \
             kernel gap was written down as the reason it could not have one — \
             the note and the text now disagree: {grew:?}"
        );
        assert!(
            maskless
                .iter()
                .any(|deployment| deployment.starts_with(family)),
            "no `{family}*` deployment is in the catalog at all, so this gate asserts \
             nothing about it"
        );
    }

    assert!(
        maskless.iter().any(|deployment| deployment == MASKLESS_RIG),
        "`{MASKLESS_RIG}` is either gone from the catalog or bakes an \
         `attention.masked` arm, and it is the artifact the maskless rig boots \
         to watch a maskless model refuse a mask — pick another row that is \
         genuinely maskless and name it here: {maskless:?}"
    );
}

mod maskless {}

mod devgeo {}

mod gemma {}
