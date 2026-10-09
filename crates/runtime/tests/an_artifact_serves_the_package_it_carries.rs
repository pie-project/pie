//! An artifact that carries its model's package is served from that package:
//! its deployment is named, composed and traced by the package, and a model
//! this build's catalog has never heard of serves all the same.

use std::collections::BTreeMap;

use checkpoint::file::write::Writer;
use checkpoint::serving::Stamp;
use poem::star::Package;
use runtime::engine::load::{Overrides, packaged};

const PACKAGE: &str = r#"
MODELS = [model("stranger-1b", template = "qwen_3", tokenizer = "qwen_3", arch = "stranger", layers = 1, vocab = 16)]
DEPLOYMENTS = [deployment("stranger-1b", weights = dtype.bf16, kv = dtype.bf16)]
"#;

const MODEL: &str = r#"
def layout(id, deploy):
    return struct(
        embed = weight("embed", [16, 8], deploy.weights[0]),
        head = weight("head", [16, 8], deploy.weights[0]),
    )
"#;

const FORWARD: &str = r#"
def caches(m, c):
    pass

def forward(m, inputs):
    return ops.linear.lm_head(ops.layout.embed(inputs.tokens(), m.embed, 16), m.head)
"#;

fn artifact(dir: &std::path::Path, package: &Package) -> std::path::PathBuf {
    let path = dir.join("stranger.zt");
    let writer = Writer::create_serving(
        &path,
        &package.attributes(),
        Stamp::of("cuda", "stranger-1b-bf16-kv-bf16"),
    )
    .expect("open the artifact");
    writer.finish().expect("finish the artifact");
    path
}

#[test]
fn a_model_no_catalog_lists_serves_from_its_artifact() {
    let package = Package::new(
        "stranger",
        &[
            ("package.poem", PACKAGE),
            ("model.poem", MODEL),
            ("forward.poem", FORWARD),
        ],
    )
    .unwrap();
    assert!(models::deployment("stranger-1b-bf16-kv-bf16").is_none());
    let dir = tempfile::tempdir().unwrap();
    let path = artifact(dir.path(), &package);

    let (name, tp, trace) = packaged(&path, &Overrides::default(), poem::Platform::Cuda)
        .unwrap()
        .expect("the artifact carries its package");
    assert_eq!((name.as_str(), tp), ("stranger-1b-bf16-kv-bf16", 1));
    assert_eq!(trace.name, name);
    assert_eq!(trace.params.len(), 2);

    let registered = runtime::model::deployment_of(&name, &path).unwrap();
    assert_eq!(
        (
            registered.entry.arch,
            registered.entry.layers,
            registered.entry.vocab
        ),
        ("stranger", 1, 16),
        "the runtime registers it by its package"
    );

    let kv = Overrides {
        kv: Some(poem::Dtype::E4m3),
        ..Overrides::default()
    };
    let (name, _, _) = packaged(&path, &kv, poem::Platform::Cuda).unwrap().unwrap();
    assert_eq!(
        name, "stranger-1b-bf16-kv-e4m3",
        "a config's kv is the package's to name"
    );

    let precision = Overrides {
        precision: Some(vec![poem::Dtype::U4g64]),
        ..Overrides::default()
    };
    let why = packaged(&path, &precision, poem::Platform::Cuda).unwrap_err();
    assert!(
        format!("{why:#}").contains("import the checkpoint at that precision"),
        "{why:#}"
    );
}

#[test]
fn an_artifact_without_a_package_is_left_to_the_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.zt");
    Writer::create_serving(&path, &BTreeMap::new(), Stamp::of("cuda", "anything"))
        .unwrap()
        .finish()
        .unwrap();
    assert!(
        packaged(&path, &Overrides::default(), poem::Platform::Cuda)
            .unwrap()
            .is_none()
    );
}
