//! An artifact is served by the package its stamp names, read from the
//! installed models directory: its deployment is named, composed and traced
//! by that package, and a model this build embeds no package for serves all
//! the same once its package is under `models/`.

use std::collections::BTreeMap;

use checkpoint::file::write::Writer;
use checkpoint::serving::Stamp;
use poem::star::Package;
use runtime::engine::load::{Overrides, packaged};

const PACKAGE: &str = r#"
CHATML = template("chatml", thinking = True, tools = True, stop = ["<|im_end|>"])
MODELS = [model("stranger-1b", template = CHATML, tokenizer = tokenizer(markers = [["<|im_end|>"]]), arch = "stranger", layers = 1, vocab = 16)]
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
        &BTreeMap::new(),
        Stamp::of("cuda", "stranger-1b-bf16-kv-bf16")
            .with_package(package.name(), &package.digest()),
    )
    .expect("open the artifact");
    writer.finish().expect("finish the artifact");
    path
}

/// The two cases share one catalog, installed once, so they run in order.
#[test]
fn an_artifact_is_served_by_the_package_its_stamp_names() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    let stranger = models.join("stranger");
    std::fs::create_dir_all(&stranger).unwrap();
    for (file, source) in [
        ("package.poem", PACKAGE),
        ("model.poem", MODEL),
        ("forward.poem", FORWARD),
    ] {
        std::fs::write(stranger.join(file), source).unwrap();
    }
    assert!(runtime::catalog::embedded().package("stranger").is_none());
    runtime::catalog::install(&models);
    a_package_under_models_serves_its_artifact();
    an_artifact_without_a_package_is_refused();
}

fn a_package_under_models_serves_its_artifact() {
    let package = Package::new(
        "stranger",
        &[
            ("package.poem", PACKAGE),
            ("model.poem", MODEL),
            ("forward.poem", FORWARD),
        ],
    )
    .unwrap();
    assert!(
        runtime::catalog::deployment("stranger-1b-bf16-kv-bf16").is_some(),
        "the installed package lists it"
    );
    let dir = tempfile::tempdir().unwrap();
    let path = artifact(dir.path(), &package);

    let (name, tp, trace) = packaged(&path, &Overrides::default(), poem::Platform::Cuda).unwrap();
    assert_eq!((name.as_str(), tp), ("stranger-1b-bf16-kv-bf16", 1));
    assert_eq!(trace.name, name);
    assert_eq!(trace.params.len(), 2);

    let registered = runtime::model::deployment_of(&name, &path).unwrap();
    assert_eq!(
        (
            registered.model.arch.as_str(),
            registered.model.layers,
            registered.model.vocab
        ),
        ("stranger", 1, 16),
        "the runtime registers it by its package"
    );

    let kv = Overrides {
        kv: Some(poem::Dtype::E4m3),
        ..Overrides::default()
    };
    let (name, _, _) = packaged(&path, &kv, poem::Platform::Cuda).unwrap();
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

fn an_artifact_without_a_package_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.zt");
    Writer::create_serving(&path, &BTreeMap::new(), Stamp::of("cuda", "anything"))
        .unwrap()
        .finish()
        .unwrap();
    let why = packaged(&path, &Overrides::default(), poem::Platform::Cuda)
        .unwrap_err()
        .to_string();
    assert!(why.contains("names no model package"), "{why}");
}
