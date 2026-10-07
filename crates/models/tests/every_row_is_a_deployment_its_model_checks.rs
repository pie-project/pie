use models::catalog::{Deploy, Drafter, Part};
use poem_dsl::{Dtype, Platform};

/// Every catalog row is one deployment of one model: the model checks it,
/// and names it by the row's name.
#[test]
fn every_row_is_a_deployment_its_model_checks() {
    let mut faults = Vec::new();
    for row in models::deployments() {
        if let Err(why) = row.entry.check(&row.deploy, Platform::Cuda) {
            faults.push(format!("`{}`: {why}", row.name));
        }
        let named = row.entry.name(&row.deploy);
        if named != row.name {
            faults.push(format!("`{}` is named `{named}` by its model", row.name));
        }
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

#[test]
fn every_model_is_one_entry() {
    let mut ids: Vec<&str> = models::entries().map(|entry| entry.id).collect();
    ids.sort_unstable();
    let len = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), len, "two entries share an id");
}

fn deploy(weights: &[Dtype], tp: u32, parts: &[Part], drafter: Option<Drafter>) -> Deploy {
    Deploy {
        weights: weights.to_vec(),
        kv: Dtype::Bf16,
        tp,
        parts: parts.to_vec(),
        drafter,
    }
}

/// A deployment no row names is still one the model checks, when the model
/// builds it and its trace splits; and refused when it does not.
#[test]
fn a_deployment_is_checked_against_its_model() {
    let qwen = models::entry("qwen36-27b").expect("the catalog ships qwen36-27b");
    qwen.check(
        &deploy(&[Dtype::Bf16], 1, &[Part::Vision], None),
        Platform::Cuda,
    )
    .expect("a bf16 qwen36-27b serves its vision tower without a drafter");
    assert!(
        qwen.check(
            &deploy(&[Dtype::Bf16], 1, &[Part::SelfCond], None),
            Platform::Cuda
        )
        .is_err()
    );
    assert!(
        qwen.check(
            &deploy(&[Dtype::Bf16], 1, &[], Some(Drafter::Eagle)),
            Platform::Cuda
        )
        .is_err()
    );

    let flash = models::entry("dsv4-flash").expect("the catalog ships dsv4-flash");
    flash
        .check(&deploy(&[Dtype::Bf16], 8, &[], None), Platform::Cuda)
        .expect("dsv4-flash splits eight ways");

    let gpt = models::entry("gptoss-20b").expect("the catalog ships gptoss-20b");
    assert!(
        gpt.check(
            &deploy(&[Dtype::Bf16, Dtype::Mxfp4], 4, &[], None),
            Platform::Cuda
        )
        .is_err(),
        "gpt-oss-20b's expert width does not split four ways"
    );
}
