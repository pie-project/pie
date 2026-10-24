//! A package's entries are found in whichever file defines each, so it may
//! be one file; a builtin called outside its stage, or another misuse of the
//! DSL, is an error naming the script's line rather than a crash.

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
        weights: vec![Dtype::Bf16.into()],
        kv: Dtype::Bf16,
        tp: 1,
        parts: vec![],
        drafter: None,
        yarn: None,
        tap: None,
    }
}

fn package(model: &str, forward: &str) -> anyhow::Result<Package> {
    Package::new("toy", &[("model.poem", model), ("forward.poem", forward)])
}

#[test]
fn a_forward_traces_the_ops_it_calls() {
    let forward = format!(
        "{CACHES}
def forward(m, inputs):
    x = ops.gather.embed(inputs.tokens(), m.embed, 16)
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
    assert_eq!(ops, ["gather.embed", "linear.matmul"], "{ops:?}");
    assert_eq!(trace.params.len(), 2);
}

#[test]
fn an_op_called_by_a_layout_is_an_error_at_its_line() {
    let model = r#"
def layout(id, deploy):
    return ops.linear.matmul(None, None)
"#;
    let forward = format!("{CACHES}\ndef forward(m, inputs):\n    return None\n");
    let why = package(model, &forward)
        .expect("every file sees every builtin, so the package loads")
        .trace("toy", &deploy(), "toy-bf16", Platform::Cuda)
        .expect_err("a layout records no trace for an op to join");
    let why = format!("{why:#}");
    assert!(why.contains("ops.linear.matmul"), "{why}");
    assert!(
        why.contains("toy/model.poem:3"),
        "at the line that calls one: {why}"
    );
}

#[test]
fn a_misused_op_is_an_error_at_the_scripts_line() {
    let forward = format!(
        "{CACHES}
def forward(m, inputs):
    x = ops.gather.embed(inputs.tokens(), m.embed, 16)
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
        why.contains("toy/forward.poem"),
        "it names the script: {why}"
    );
}

#[test]
fn a_package_may_be_one_file() {
    let whole = format!(
        "{MODEL}\n{CACHES}\ndef forward(m, inputs):\n    return ops.linear.lm_head(ops.gather.embed(inputs.tokens(), m.embed, 16), m.head)\n"
    );
    let package = Package::new("toy", &[("toy.poem", &whole)]).expect("one file, every entry");
    package
        .trace("toy", &deploy(), "toy-bf16", Platform::Cuda)
        .expect("it traces as the files apart do");
}

#[test]
fn an_entry_is_defined_once() {
    let forward = format!("{MODEL}\n{CACHES}\ndef forward(m, inputs):\n    return None\n");
    let why = package(MODEL, &forward)
        .expect("a package loads whatever its files define")
        .trace("toy", &deploy(), "toy-bf16", Platform::Cuda)
        .expect_err("`layout` is defined twice");
    let why = format!("{why:#}");
    assert!(
        why.contains("defines `layout` in both `forward.poem` and `model.poem`"),
        "{why}"
    );
}

#[test]
fn a_library_is_loaded_by_any_file_that_needs_it() {
    let forward = r#"
load("//lib/head/forward.poem", "read_out")

def caches(m, c):
    pass

def forward(m, inputs):
    return read_out(m, ops.gather.embed(inputs.tokens(), m.embed, 16))
"#;
    let library = r#"
def read_out(m, x):
    return ops.linear.lm_head(x, m.head)
"#;
    let package = Package::new(
        "toy",
        &[
            ("model.poem", MODEL),
            ("forward.poem", forward),
            ("//lib/head/forward.poem", library),
        ],
    )
    .unwrap_or_else(|e| panic!("{e:#}"));
    let trace = package
        .trace("toy", &deploy(), "toy-bf16", Platform::Cuda)
        .unwrap_or_else(|e| panic!("{e:#}"));
    assert_eq!(trace.params.len(), 2);

    // A file that loads a helper binds it without defining it: the helper's
    // entries stay the helper's own.
    let layout = "load(\"//lib/head/forward.poem\", \"read_out\")\n".to_string() + MODEL;
    Package::new(
        "toy",
        &[
            ("model.poem", &layout),
            ("forward.poem", forward),
            ("//lib/head/forward.poem", library),
        ],
    )
    .unwrap_or_else(|e| panic!("{e:#}"))
    .trace("toy", &deploy(), "toy-bf16", Platform::Cuda)
    .unwrap_or_else(|e| panic!("two files load one helper: {e:#}"));
}
