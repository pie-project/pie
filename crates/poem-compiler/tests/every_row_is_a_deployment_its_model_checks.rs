use poem::star::Deploy;
use poem::{Dtype, Platform};

/// Every catalog row is one deployment of one model: the model checks it,
/// and names it by the row's name.
#[test]
fn every_row_is_a_deployment_its_model_checks() {
    let mut faults = Vec::new();
    for row in poem_compiler::catalog::deployments() {
        if let Err(why) = row.check(Platform::Cuda) {
            faults.push(format!("`{}`: {why}", row.name));
        }
        let named = row.model.name(&row.deploy);
        if named != row.name {
            faults.push(format!("`{}` is named `{named}` by its model", row.name));
        }
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

#[test]
fn every_model_is_one_entry() {
    let mut ids: Vec<&str> = poem_compiler::catalog::repository()
        .models()
        .map(|(_, model)| model.id.as_str())
        .collect();
    ids.sort_unstable();
    let len = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), len, "two entries share an id");
}

fn deploy(weights: &[Dtype], tp: u32, parts: &[&str], drafter: Option<&str>) -> Deploy {
    Deploy {
        weights: weights.to_vec(),
        kv: Dtype::Bf16,
        tp,
        parts: parts.iter().map(|p| (*p).to_string()).collect(),
        drafter: drafter.map(str::to_string),
    }
}

/// Whether the model `id` serves `deploy` on `platform`.
fn check(id: &str, deploy: Deploy, platform: Platform) -> Result<(), String> {
    let (package, model) = poem_compiler::catalog::repository()
        .model(id)
        .unwrap_or_else(|| panic!("the catalog ships {id}"));
    poem_compiler::catalog::Deployment::of(package.clone(), model, deploy)
        .check(platform)
        .map_err(|why| why.0)
}

/// A deployment no row names is still one the model checks, when the model
/// builds it and its trace splits; and refused when it does not.
#[test]
fn a_deployment_is_checked_against_its_model() {
    check(
        "qwen36-27b",
        deploy(&[Dtype::Bf16], 1, &["vision"], None),
        Platform::Cuda,
    )
    .expect("a bf16 qwen36-27b serves its vision tower without a drafter");
    assert!(
        check(
            "qwen36-27b",
            deploy(&[Dtype::Bf16], 1, &["selfcond"], None),
            Platform::Cuda
        )
        .is_err()
    );
    assert!(
        check(
            "qwen36-27b",
            deploy(&[Dtype::Bf16], 1, &[], Some("eagle")),
            Platform::Cuda
        )
        .is_err()
    );

    check(
        "dsv4-flash",
        deploy(&[Dtype::Bf16], 8, &[], None),
        Platform::Cuda,
    )
    .expect("dsv4-flash splits eight ways");

    assert!(
        check(
            "gptoss-20b",
            deploy(&[Dtype::Bf16, Dtype::Mxfp4], 4, &[], None),
            Platform::Cuda
        )
        .is_err(),
        "gpt-oss-20b's expert width does not split four ways"
    );
}
