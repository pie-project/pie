//! A package's files see only their stage's builtins, a misuse of the DSL is
//! an error naming the script's line rather than a crash, and a stage loads
//! no other stage.

use poem::star::{Deploy, Package};
use poem::{Dtype, Platform};

const MODEL: &str = r#"
def layout(id, deploy):
    return struct(
        embed = weight("embed", [16, 8], deploy.weights[0]),
        head = weight("head", [16, 8], deploy.weights[0]),
    )
"#;

const CACHES: &str = r#"
def caches(m, c):
    pass
"#;

fn deploy() -> Deploy {
    Deploy {
        weights: vec![Dtype::Bf16],
        kv: Dtype::Bf16,
        tp: 1,
        parts: vec![],
        drafter: None,
    }
}

fn package(model: &str, forward: &str) -> anyhow::Result<Package> {
    Package::new("toy", &[("model.star", model), ("forward.star", forward)])
}

#[test]
fn a_forward_traces_the_ops_it_calls() {
    let forward = format!(
        "{CACHES}
def forward(m, inputs):
    x = ops.layout.embed(inputs.tokens(), m.embed, 16)
    return ops.linear.lm_head(x, m.head)
"
    );
    let trace = package(MODEL, &forward)
        .unwrap()
        .trace("toy", &deploy(), "toy-bf16", Platform::Cuda)
        .unwrap_or_else(|e| panic!("{e:#}"));
    let ops: Vec<&str> = trace
        .nodes
        .iter()
        .map(|n| {
            use poem::Operands;
            n.op.name()
        })
        .collect();
    assert_eq!(ops, ["layout.embed", "linear.lm_head"], "{ops:?}");
    assert_eq!(trace.params.len(), 2);
}

#[test]
fn a_layout_cannot_call_an_op() {
    let model = r#"
def layout(id, deploy):
    return ops.linear.matmul(None, None)
"#;
    let forward = format!("{CACHES}\ndef forward(m, inputs):\n    return None\n");
    let why = package(model, &forward)
        .err()
        .expect("a layout has no ops, so the package does not load");
    let why = format!("{why:#}");
    assert!(why.contains("Variable `ops` not found"), "{why}");
    assert!(
        why.contains("toy/model.star:3"),
        "at the line that calls one: {why}"
    );
}

#[test]
fn a_misused_op_is_an_error_at_the_scripts_line() {
    let forward = format!(
        "{CACHES}
def forward(m, inputs):
    x = ops.layout.embed(inputs.tokens(), m.embed, 16)
    y = ops.elemwise.residual_add(x.on(fact.drafts()), x.on(~fact.drafts()))
    return ops.linear.lm_head(y, m.head)
"
    );
    let why = package(MODEL, &forward)
        .unwrap()
        .trace("toy", &deploy(), "toy-bf16", Platform::Cuda)
        .expect_err("two arms do not add");
    let why = format!("{why:#}");
    assert!(
        why.contains("mixes values from different split arms"),
        "{why}"
    );
    assert!(
        why.contains("toy/forward.star"),
        "it names the script: {why}"
    );
}

#[test]
fn a_stage_loads_no_other_stage() {
    let forward = format!(
        "load(\"model.star\", \"layout\")\n{CACHES}\ndef forward(m, inputs):\n    return None\n"
    );
    let why = package(MODEL, &forward)
        .err()
        .expect("a forward does not load the layout's file");
    assert!(
        format!("{why:#}").contains("a stage loads only helper files"),
        "{why:#}"
    );
}

#[test]
fn a_library_is_loaded_by_the_files_of_its_own_stage() {
    let forward = r#"
load("//lib/head/forward.star", "read_out")

def caches(m, c):
    pass

def forward(m, inputs):
    return read_out(m, ops.layout.embed(inputs.tokens(), m.embed, 16))
"#;
    let library = r#"
def read_out(m, x):
    return ops.linear.lm_head(x, m.head)
"#;
    let package = Package::new(
        "toy",
        &[
            ("model.star", MODEL),
            ("forward.star", forward),
            ("//lib/head/forward.star", library),
        ],
    )
    .unwrap_or_else(|e| panic!("{e:#}"));
    let trace = package
        .trace("toy", &deploy(), "toy-bf16", Platform::Cuda)
        .unwrap_or_else(|e| panic!("{e:#}"));
    assert_eq!(trace.params.len(), 2);

    let layout = "load(\"//lib/head/forward.star\", \"read_out\")\n".to_string() + MODEL;
    let why = Package::new(
        "toy",
        &[
            ("model.star", &layout),
            ("//lib/head/forward.star", library),
        ],
    )
    .err()
    .map(|e| format!("{e:#}"))
    .expect("a layout loads no forward library");
    assert!(why.contains("of its own stage"), "{why}");
}
