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
    a_step_module_writes_only_its_heap_and_its_status();
}

/// Fails the check with its message where `cond` does not hold.
macro_rules! check {
    ($cond:expr, $($why:tt)+) => {
        if !$cond {
            return Err(format!($($why)+));
        }
    };
}

/// What a step can write, store by store (the runtime's isolation facts):
/// its heap only through `hst`, at `b_heap + w` (w checked against the
/// heap's words there); its status only through `pie_fault`, at `b_status`
/// and a constant; its tables and its record nowhere. The bases and the heap
/// bound are written only at the entry, from the record and the config
/// table. A backend may then bind the tables read_write beside the status
/// (wgpu's usage scope) and lose nothing of a read-only binding's guarantee.
/// Each way out is refused (the negative controls).
fn a_step_module_writes_only_its_heap_and_its_status() {
    for call in step_calls() {
        let source = step_module("main", &call);
        if let Err(why) = writes_only_its_heap_and_its_status(&call, &source) {
            panic!("{why}");
        }
    }
    let call = &step_calls()[0];
    let source = step_module("main", call);
    for (planted, refused) in [
        ("fn planted() { heap[3u] = 1u; }", "stores to the heap"),
        (
            "fn planted() { heap[b_heap + 1u] = 1u; }",
            "stores to the heap",
        ),
        (
            "fn planted() { atomicStore(&status[0u], 1u); }",
            "stores to the status",
        ),
        ("fn planted() { b_heap = 0u; }", "sets `b_heap`"),
        ("fn planted() { lim_heap = 4096u; }", "sets `lim_heap`"),
    ] {
        let source = format!(
            "{source}
{planted}
"
        );
        let why = writes_only_its_heap_and_its_status(call, &source).expect_err(planted);
        assert!(
            why.contains(refused),
            "`{planted}` refused as `{why}`, not `{refused}`"
        );
    }
}

fn writes_only_its_heap_and_its_status(call: &str, source: &str) -> Result<(), String> {
    const READ_ONLY: [&str; 2] = ["tables", "pie_step"];
    const SET_AT_ENTRY: [&str; 7] = [
        "b_status",
        "b_descs",
        "b_params",
        "b_offs",
        "b_heap",
        "lim_heap",
        "tmp_base",
    ];
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    let named = |name: &str| {
        module
            .global_variables
            .iter()
            .find(|(_, var)| var.name.as_deref() == Some(name))
            .map(|(handle, _)| handle)
            .ok_or_else(|| format!("call {call} declares no `{name}`"))
    };
    let (status, heap) = (named("status")?, named("heap")?);
    let (b_status, b_heap) = (named("b_status")?, named("b_heap")?);
    let read_only = READ_ONLY
        .iter()
        .map(|name| named(name))
        .collect::<Result<Vec<_>, _>>()?;
    let set_at_entry = SET_AT_ENTRY
        .iter()
        .map(|name| named(name))
        .collect::<Result<Vec<_>, _>>()?;
    let functions = module
        .functions
        .iter()
        .map(|(_, function)| (function, false))
        .chain(
            module
                .entry_points
                .iter()
                .map(|entry| (&entry.function, true)),
        );
    for (function, entry) in functions {
        let name = function.name.as_deref().unwrap_or("?");
        let exprs = &function.expressions;
        let mut stores = Vec::new();
        stores_in(&function.body, &mut stores);
        for (pointer, value) in stores {
            let Root::Global(global) = root_of(exprs, pointer) else {
                check!(
                    !matches!(root_of(exprs, pointer), Root::Argument),
                    "call {call}: `{name}` stores through a pointer argument, which this \
                     test cannot follow"
                );
                continue;
            };
            let global_name = module.global_variables[global]
                .name
                .as_deref()
                .unwrap_or("?");
            check!(
                !read_only.contains(&global),
                "call {call}: `{name}` stores to `{global_name}`"
            );
            if global == status {
                check!(
                    name == "pie_fault",
                    "call {call}: `{name}` stores to the status"
                );
                let index = index_of(exprs, pointer).ok_or("a status store indexes no word")?;
                check!(
                    based(exprs, index, b_status, |e| matches!(
                        e,
                        naga::Expression::Literal(_)
                    )),
                    "call {call}: a status store not at `b_status` and a constant"
                );
            } else if global == heap {
                check!(name == "hst", "call {call}: `{name}` stores to the heap");
                let index = index_of(exprs, pointer).ok_or("a heap store indexes no word")?;
                check!(
                    based(exprs, index, b_heap, |_| true),
                    "call {call}: a heap store not at `b_heap` and its word"
                );
            } else if set_at_entry.contains(&global) {
                check!(entry, "call {call}: `{name}` sets `{global_name}`");
                let read = value
                    .and_then(|value| match exprs[value] {
                        naga::Expression::Load { pointer } => Some(root_of(exprs, pointer)),
                        _ => None,
                    })
                    .and_then(|root| match root {
                        Root::Global(read) => Some(read),
                        _ => None,
                    });
                check!(
                    read.is_some_and(|read| read_only.contains(&read)),
                    "call {call}: `{global_name}` set from other than the record or the \
                     config table"
                );
            }
        }
    }
    Ok(())
}

/// The index a store's pointer takes into its global, where it is one.
fn index_of(
    exprs: &naga::Arena<naga::Expression>,
    pointer: naga::Handle<naga::Expression>,
) -> Option<naga::Handle<naga::Expression>> {
    match exprs[pointer] {
        naga::Expression::Access { index, .. } => Some(index),
        _ => None,
    }
}

/// Whether `index` is the base `base` read, alone or plus a term `rest`
/// takes.
fn based(
    exprs: &naga::Arena<naga::Expression>,
    index: naga::Handle<naga::Expression>,
    base: naga::Handle<naga::GlobalVariable>,
    rest: impl Fn(&naga::Expression) -> bool,
) -> bool {
    let reads_base = |at: naga::Handle<naga::Expression>| {
        matches!(exprs[at], naga::Expression::Load { pointer }
            if matches!(exprs[pointer], naga::Expression::GlobalVariable(global) if global == base))
    };
    match exprs[index] {
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left,
            right,
        } => reads_base(left) && rest(&exprs[right]),
        _ => reads_base(index),
    }
}

type Stored = (
    naga::Handle<naga::Expression>,
    Option<naga::Handle<naga::Expression>>,
);

fn stores_in(block: &naga::Block, out: &mut Vec<Stored>) {
    for statement in block.iter() {
        match statement {
            naga::Statement::Store { pointer, value } => out.push((*pointer, Some(*value))),
            naga::Statement::Atomic { pointer, .. } => out.push((*pointer, None)),
            naga::Statement::Block(inner) => stores_in(inner, out),
            naga::Statement::If { accept, reject, .. } => {
                stores_in(accept, out);
                stores_in(reject, out);
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                stores_in(body, out);
                stores_in(continuing, out);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    stores_in(&case.body, out);
                }
            }
            _ => {}
        }
    }
}

enum Root {
    Global(naga::Handle<naga::GlobalVariable>),
    Argument,
    Local,
}

fn root_of(
    expressions: &naga::Arena<naga::Expression>,
    mut at: naga::Handle<naga::Expression>,
) -> Root {
    loop {
        match expressions[at] {
            naga::Expression::GlobalVariable(global) => return Root::Global(global),
            naga::Expression::Access { base, .. } | naga::Expression::AccessIndex { base, .. } => {
                at = base;
            }
            naga::Expression::FunctionArgument(_) => return Root::Argument,
            _ => return Root::Local,
        }
    }
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
        "@group(0) @binding(1) var<storage, read>       tables",
        "@group(0) @binding(4) var<storage, read_write> heap",
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
