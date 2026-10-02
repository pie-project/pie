use model_dsl::Platform;
use model_ir::{Attention, Operation, Trace};

fn carries_a_head(sku: &str) -> bool {
    models::published::PUBLISHED.iter().any(|p| p.sku == sku)
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
    for row in models::skus() {
        let (sku, trace) = (row.name.as_str(), row.trace);
        let arms = masked_arms(&trace(Platform::Cuda));
        if arms > 0 {
            declaring.push((sku.to_string(), arms));
        } else {
            maskless.push(sku.to_string());
        }
    }

    assert!(
        declaring.iter().all(|(sku, _)| {
            DECLARE.iter().any(|family| sku.starts_with(family)) || carries_a_head(sku)
        }),
        "a family beyond gemma and qwen declares `attention.masked` from its \
         own text — not from an overlaid drafter head — and the device gates \
         in this file were written against gemma: {declaring:?}"
    );
    assert!(
        !declaring.is_empty(),
        "no SKU declares `attention.masked` at all, and then the axis has no \
         model text to be exercised by"
    );

    for family in DECLARE {
        assert!(
            declaring.iter().any(|(sku, _)| sku.starts_with(family)),
            "no `{family}*` SKU declares `attention.masked` any more, so the \
             axis lost a family: {declaring:?}"
        );
    }

    for family in GAPPED {
        let grew: Vec<&(String, usize)> = declaring
            .iter()
            .filter(|(sku, _)| sku.starts_with(family) && !carries_a_head(sku))
            .collect();
        assert!(
            grew.is_empty(),
            "`{family}*` grew an `attention.masked` arm in its own text, and a \
             kernel gap was written down as the reason it could not have one — \
             the note and the text now disagree: {grew:?}"
        );
        assert!(
            maskless.iter().any(|sku| sku.starts_with(family)),
            "no `{family}*` SKU is in the catalog at all, so this gate asserts \
             nothing about it"
        );
    }

    assert!(
        maskless.iter().any(|sku| sku == MASKLESS_RIG),
        "`{MASKLESS_RIG}` is either gone from the catalog or bakes an \
         `attention.masked` arm, and it is the artifact the maskless rig boots \
         to watch a maskless model refuse a mask — pick another row that is \
         genuinely maskless and name it here: {maskless:?}"
    );
}

mod maskless {}

mod devgeo {}

mod gemma {}
