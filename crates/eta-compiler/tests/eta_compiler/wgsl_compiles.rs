use eta_compiler::codegen::launch::{LaunchOp, LaunchPlanValue, LaunchStagePlan};
use eta_compiler::codegen::wgsl::{RUNTIME, WORKGROUP, emit_launch_steps};
use eta_compiler::plan::Dimension;
use eta_ir::op::tags;
use eta_ir::types::Dtype;

fn compile(source: &str) -> Vec<u32> {
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(module) => module,
        Err(error) => panic!(
            "the emitted WGSL does not parse:\n{}",
            error.emit_to_string(source)
        ),
    };
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("the emitted WGSL does not validate: {error:?}"));
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .expect("the validated module lowers to SPIR-V")
}

fn body(calls: &str) -> String {
    format!(
        "{RUNTIME}\n@compute @workgroup_size({WORKGROUP})\n\
         fn guest_pass(@builtin(local_invocation_id) lid : vec3<u32>) {{\n  \
         tid = lid.x;\n  lanes = {WORKGROUP}u;\n{calls}}}\n"
    )
}

#[test]
fn wgsl_compiles_every_case() {
    the_runtime_compiles();
    a_body_of_ordinary_ops_compiles();
    a_reduce_ladder_compiles();
    a_sort_ladder_compiles();
    the_shader_declares_the_bindings_a_shell_binds();
    a_stepwise_module_compiles();
}

fn the_runtime_compiles() {
    let words = compile(&body(""));
    assert_eq!(words[0], 0x0723_0203, "the first word is the SPIR-V magic");
    assert!(
        words.len() > 1000,
        "a runtime this size lowers to more than {} words",
        words.len()
    );
}

fn a_body_of_ordinary_ops_compiles() {
    let calls = (0..8)
        .map(|node| format!("  ptir_step({node}u);\n  storageBarrier();\n"))
        .collect::<String>();
    compile(&body(&calls));
}

fn a_reduce_ladder_compiles() {
    let mut calls = String::new();
    for level in 0..7 {
        calls.push_str(&format!(
            "  ptir_reduce_level(0u, {level}u);\n  storageBarrier();\n"
        ));
    }
    calls.push_str("  ptir_reduce_finish(0u);\n  storageBarrier();\n");
    compile(&body(&calls));
}

fn a_sort_ladder_compiles() {
    let mut calls = String::from("  ptir_sort_seed(0u);\n  storageBarrier();\n");
    for round in 0..28 {
        calls.push_str(&format!(
            "  ptir_sort_round(0u, {round}u);\n  storageBarrier();\n"
        ));
    }
    calls.push_str("  ptir_sort_pre(0u);\n  storageBarrier();\n");
    calls.push_str("  ptir_step(0u);\n  storageBarrier();\n");
    compile(&body(&calls));
}

fn the_shader_declares_the_bindings_a_shell_binds() {
    for declaration in [
        "@group(0) @binding(0) var<storage, read_write> status",
        "@group(0) @binding(1) var<storage, read>       descs",
        "@group(0) @binding(2) var<storage, read>       params",
        "@group(0) @binding(3) var<storage, read>       offs",
        "@group(0) @binding(4) var<storage, read_write> heap",
        "@group(0) @binding(5) var<uniform>             cfg",
    ] {
        assert!(
            RUNTIME.contains(declaration),
            "the runtime no longer declares `{declaration}`"
        );
    }
}

fn a_stepwise_module_compiles() {
    let dim = |n| LaunchPlanValue {
        dtype: Dtype::F32,
        axes: alloc_axes(n),
    };
    let op = |tag, result_id, args: &[u32]| LaunchOp {
        tag,
        result_count: 1,
        result_id,
        args: args.to_vec(),
        ..LaunchOp::default()
    };
    let plan = LaunchStagePlan {
        ops: vec![
            op(tags::IOTA, 0, &[]),
            op(tags::EXP, 1, &[0]),
            op(tags::REDUCE_SUM, 2, &[1]),
            op(tags::SORT_DESC, 3, &[1]),
        ],
        value_types: vec![dim(32_768), dim(32_768), dim(1), dim(32_768)],
        ..LaunchStagePlan::default()
    };
    let stepwise = emit_launch_steps("guest_pass", &plan).expect("the stepwise shape emits");

    // iota, exp, a reduce of three 32-fold levels and its finish, and the
    // sort's seed, 16 rounds, pre, finish and pack.
    assert_eq!(stepwise.steps.len(), 26, "every rung is its own dispatch");

    let words = compile(&stepwise.source);
    assert_eq!(words[0], 0x0723_0203, "the first word is the SPIR-V magic");

    for step in &stepwise.steps {
        assert!(
            stepwise
                .source
                .contains(&format!("fn {}(@builtin(global_invocation_id)", step.entry)),
            "step `{}` names an entry point the module does not declare",
            step.entry
        );
    }
}

fn alloc_axes(n: u32) -> Vec<Dimension> {
    vec![Dimension::Static(n)]
}
