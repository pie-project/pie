//! `package.poem` states a package's models and the deployments it lists,
//! each named by the one grammar every deployment is named by; and the
//! package travels in an artifact's attributes, refused under other builtins.

use poem::Platform;
use poem::star::{API, ATTRIBUTE, Deploy, Package};

const PACKAGE: &str = r#"
TOY = template("raw", stop = ["<eos>"])
TOKENIZER = tokenizer(markers = [["<eos>"]], parts = {"vision": [["<image>"]]})
MODELS = [
    model("toy-1b", template = TOY, tokenizer = TOKENIZER, parts = ["vision"], drafters = ["mtp"]),
    model("toy-1b-mini-l2", mini = True, template = TOY, tokenizer = TOKENIZER),
]

DEPLOYMENTS = [
    deployment("toy-1b", weights = dtype.bf16, kv = dtype.bf16),
    deployment("toy-1b", weights = [dtype.u4g64, dtype.u2g64], kv = dtype.bf16, parts = ["vision"], drafter = "mtp"),
    deployment("toy-1b-mini-l2", weights = dtype.bf16, kv = dtype.e4m3),
]
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

fn toy() -> Package {
    Package::new(
        "toy",
        &[
            ("package.poem", PACKAGE),
            ("model.poem", MODEL),
            ("forward.poem", FORWARD),
        ],
    )
    .unwrap_or_else(|e| panic!("{e:#}"))
}

#[test]
fn a_manifest_lists_its_models_and_deployments_in_order() {
    let package = toy();
    let manifest = package.manifest();
    let ids: Vec<&str> = manifest.models.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, ["toy-1b", "toy-1b-mini-l2"]);
    assert!(manifest.model("toy-1b-mini-l2").unwrap().mini);
    let names: Vec<String> = manifest
        .deployments
        .iter()
        .map(|(id, d)| manifest.model(id).unwrap().name(d))
        .collect();
    assert_eq!(
        names,
        [
            "toy-1b-bf16-kv-bf16",
            "toy-1b-vision-mtp-u4g64-u2g64-kv-bf16",
            "toy-1b-mini-l2-bf16-kv-e4m3",
        ]
    );
}

#[test]
fn a_name_parses_back_to_the_deployment_it_names() {
    let package = toy();
    let manifest = package.manifest();
    for (id, deploy) in &manifest.deployments {
        let model = manifest.model(id).unwrap();
        for tp in [1, 4] {
            let deploy = Deploy {
                tp,
                ..deploy.clone()
            };
            let name = model.name(&deploy);
            let (parsed, back) = manifest.parse(&name).expect(&name);
            assert_eq!(
                (parsed.id.as_str(), &back),
                (id.as_str(), &deploy),
                "{name}"
            );
        }
    }
    // The miniature's id extends the whole model's: the longer id is tried
    // first, so its names are not read as the whole model's.
    assert_eq!(
        manifest.parse("toy-1b-mini-l2-bf16-kv-bf16").unwrap().0.id,
        "toy-1b-mini-l2"
    );
    assert!(manifest.parse("toy-1b-audio-bf16-kv-bf16").is_none());
    assert!(manifest.parse("toy-1b-mini-l2-mtp-bf16-kv-bf16").is_none());
}

#[test]
fn a_manifest_refuses_a_deployment_its_model_cannot_serve() {
    let refused = |package: &str| {
        Package::new("toy", &[("package.poem", package)])
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_else(|| panic!("loaded:\n{package}"))
    };
    let why = refused(
        r#"
TOY = template("raw", stop = ["<eos>"])
TOKENIZER = tokenizer()
MODELS = [model("toy-1b", template = TOY, tokenizer = TOKENIZER)]
DEPLOYMENTS = [deployment("toy-1b", weights = dtype.bf16, kv = dtype.bf16, drafter = "mtp")]
"#,
    );
    assert!(why.contains("pairs with no mtp drafter"), "{why}");
    let why = refused(
        r#"
TOY = template("raw", stop = ["<eos>"])
TOKENIZER = tokenizer()
MODELS = [model("toy-1b", template = TOY, tokenizer = TOKENIZER)]
DEPLOYMENTS = [deployment("toy-2b", weights = dtype.bf16, kv = dtype.bf16)]
"#,
    );
    assert!(why.contains("states no model of"), "{why}");
}

#[test]
fn a_package_travels_in_attributes_and_traces_the_same() {
    let package = toy();
    let attributes = package.attributes();
    assert!(attributes.keys().all(|k| k.starts_with(ATTRIBUTE)));
    let carried = Package::from_attributes(
        attributes
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .chain([("pie_source", "elsewhere")]),
    )
    .unwrap()
    .expect("the attributes carry a package");
    assert_eq!(carried.attributes(), attributes);
    let (id, deploy) = &package.manifest().deployments[0];
    let trace = |p: &Package| {
        p.trace(id, deploy, "toy-1b-bf16-kv-bf16", Platform::Cuda)
            .unwrap()
    };
    assert!(trace(&carried) == trace(&package));
    assert!(
        Package::from_attributes([("pie_source", "elsewhere")])
            .unwrap()
            .is_none()
    );
}

#[test]
fn a_package_written_against_other_builtins_is_refused() {
    let mut attributes = toy().attributes();
    attributes.insert(format!("{ATTRIBUTE}api"), (API + 1).to_string());
    let why = Package::from_attributes(attributes.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .err()
        .expect("another version's package is refused");
    assert!(
        format!("{why:#}").contains("import the checkpoint again"),
        "{why:#}"
    );
}
