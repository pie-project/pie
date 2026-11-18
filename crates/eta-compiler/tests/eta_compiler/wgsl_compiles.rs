use eta_compiler::codegen::launch::{LaunchOp, LaunchPlanValue, LaunchStagePlan};
use eta_compiler::codegen::wgsl::{RUNTIME, WORKGROUP, emit_launch_steps, step_calls, step_module};
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

    // Every call of the closed set is a module of its own, and each compiles.
    let calls = step_calls();
    for (at, call) in calls.iter().enumerate() {
        let words = compile(&step_module("guest_pass", call));
        assert_eq!(words[0], 0x0723_0203, "call {at} lowers to SPIR-V");
    }
    for step in &stepwise.steps {
        assert!(
            (step.call as usize) < calls.len(),
            "step `{}` names call {}, past the {} listed",
            step.entry,
            step.call,
            calls.len()
        );
    }
}

fn alloc_axes(n: u32) -> Vec<Dimension> {
    vec![Dimension::Static(n)]
}

/// The functions of `source` that return a value and whose body's last
/// statement is not a `return`, by name. A compiler that cannot see that
/// a `switch` whose every case returns covers every path (FXC, through
/// which wgpu compiles naga's HLSL on DX12) refuses such a function ("not
/// all control paths return a value"), so every one the WGSL emits ends
/// on a return reached after its switch or if chain.
fn ends_without_a_return(source: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = source;
    while let Some(at) = rest.find("fn ") {
        let after = &rest[at + 3..];
        let Some(open) = after.find('{') else { break };
        let head = &after[..open];
        let name = head.split('(').next().unwrap_or("").trim().to_owned();
        let returns = head.contains("->");
        let mut depth = 0usize;
        let mut close = None;
        for (i, c) in after[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else { break };
        if returns {
            let body = after[open + 1..close].trim_end();
            let last = body.strip_suffix(';').map(|inner| {
                let start = inner
                    .rfind(|c| matches!(c, ';' | '{' | '}'))
                    .map_or(0, |at| at + 1);
                inner[start..].trim()
            });
            if !last.is_some_and(|statement| statement.starts_with("return")) {
                found.push(name);
            }
        }
        rest = &after[close + 1..];
    }
    found
}

#[test]
fn every_value_returning_function_ends_on_a_return() {
    let mut modules: Vec<String> = step_calls()
        .iter()
        .map(|call| step_module("main", call))
        .collect();
    modules.push(RUNTIME.to_owned());
    for module in &modules {
        let bad = ends_without_a_return(module);
        assert!(
            bad.is_empty(),
            "functions {bad:?} return a value and end on no return statement"
        );
    }
    // The scan itself: a final switch is caught, a final return is not.
    assert_eq!(
        ends_without_a_return(
            "fn a(w : u32) -> u32 { switch (w) { case 0u: { return 1u; } default: { return 2u; } } }\n\
             fn b(w : u32) -> u32 { switch (w) { case 0u: { return 1u; } default: {} } return 2u; }\n\
             fn c(y : u32) -> u32 { if (y == 0u) { return 0u; } return 1u / y; }\n\
             fn d(y : u32) { if (y == 0u) { return; } }"
        ),
        ["a"]
    );
}
