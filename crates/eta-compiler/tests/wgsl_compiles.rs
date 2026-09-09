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
    the_key_order_is_the_hosts_order();
    the_merge_ladder_orders_every_row();
    the_shader_declares_the_bindings_a_shell_binds();
    a_stepwise_module_compiles();
    the_top_p_walk_is_the_hosts_walk();
    the_monotone_guard_is_the_no_negative_lane_predicate();
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

fn sort_key(x: f32) -> u32 {
    if x.is_nan() {
        return 0xFFFF_FFFF;
    }
    let b = if x == 0.0 { 0u32 } else { x.to_bits() };
    let asc = if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    };
    0xFFFF_FFFF - asc
}

fn desc_by_value(row: &[f32], a: u32, b: u32) -> core::cmp::Ordering {
    let (x, y) = (row[a as usize], row[b as usize]);
    match (x.is_nan(), y.is_nan()) {
        (true, false) => core::cmp::Ordering::Greater,
        (false, true) => core::cmp::Ordering::Less,
        (true, true) => a.cmp(&b),
        (false, false) => {
            if x == y {
                a.cmp(&b)
            } else {
                y.partial_cmp(&x).unwrap_or(core::cmp::Ordering::Equal)
            }
        }
    }
}

fn the_key_order_is_the_hosts_order() {
    let row: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.5,
        -0.5,
        f32::NAN,
        1.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        0.0,
    ];

    let mut by_host: Vec<u32> = (0..row.len() as u32).collect();
    by_host.sort_by(|&a, &b| desc_by_value(&row, a, b));

    let mut by_key: Vec<u32> = (0..row.len() as u32).collect();
    by_key.sort_by_key(|&i| (sort_key(row[i as usize]), i));

    assert_eq!(
        by_key,
        by_host,
        "the key order is not the host's; keys {:#x?}",
        row.iter().copied().map(sort_key).collect::<Vec<_>>()
    );

    for &x in &row {
        if !x.is_nan() {
            assert!(
                sort_key(x) < 0xFFFF_FFFF,
                "{x} takes the key NaN is meant to own"
            );
        }
    }
}

fn merge_ladder(values: &[f32], len: usize, rounds: u32) -> Vec<u32> {
    let n = values.len();
    let mut a: Vec<(u32, u32)> = (0..n)
        .map(|i| (sort_key(values[i]), (i % len.max(1)) as u32))
        .collect();
    let mut b: Vec<(u32, u32)> = vec![(0, 0); n];
    for r in 0..rounds {
        let run = 1usize << r;
        if run >= len {
            b.copy_from_slice(&a);
        } else {
            let span = run * 2;
            for i in 0..n {
                let row = i / len.max(1);
                let rowb = row * len;
                let at = i - rowb;
                let blk = (at / span) * span;
                let mid = (blk + run).min(len);
                let end = (blk + span).min(len);
                let me = a[i];
                let (mut lo, mut hi, own) = if at < mid {
                    (0usize, end - mid, at - blk)
                } else {
                    (0usize, mid - blk, at - mid)
                };
                while lo < hi {
                    let m = lo + (hi - lo) / 2;
                    let c = if at < mid {
                        rowb + mid + m
                    } else {
                        rowb + blk + m
                    };
                    if a[c] < me {
                        lo = m + 1;
                    } else {
                        hi = m;
                    }
                }
                b[rowb + blk + own + lo] = me;
            }
        }
        core::mem::swap(&mut a, &mut b);
    }
    a.iter().map(|&(_, i)| i).collect()
}

fn the_merge_ladder_orders_every_row() {
    for &len in &[1usize, 2, 3, 5, 8, 9, 17, 32, 33, 64] {
        for &rows in &[1usize, 3] {
            let n = rows * len;
            let values: Vec<f32> = (0..n)
                .map(|i| match i % 7 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::NAN,
                    3 => (i % 5) as f32,
                    4 => -((i % 3) as f32),
                    5 => f32::INFINITY,
                    _ => ((i * 37 % 11) as f32) * 0.25,
                })
                .collect();

            let got = merge_ladder(&values, len, 28);

            for row in 0..rows {
                let slice = &values[row * len..row * len + len];
                let mut want: Vec<u32> = (0..len as u32).collect();
                want.sort_by(|&a, &b| desc_by_value(slice, a, b));
                assert_eq!(
                    &got[row * len..row * len + len],
                    &want[..],
                    "row {row} of {rows} at len {len} is not the host's order"
                );
            }
        }
    }
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

    assert_eq!(stepwise.steps.len(), 30, "every rung is its own dispatch");

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

#[allow(clippy::neg_cmp_op_on_partial_ord, clippy::eq_op)]
fn top_p_walk(values: &[f32], len: usize, cut: f32) -> Vec<bool> {
    const UNROLL: usize = 32;
    let rows = values.len() / len;
    let mut keep = vec![false; values.len()];
    for row in 0..rows {
        let rowb = row * len;
        let slice = &values[rowb..rowb + len];
        let mut order: Vec<u32> = (0..len as u32).collect();
        order.sort_by(|&a, &b| desc_by_value(slice, a, b));

        let vals: Vec<f32> = order.iter().map(|&i| slice[i as usize]).collect();
        let mut pos = vec![0usize; len];
        for (t, &i) in order.iter().enumerate() {
            pos[i as usize] = t;
        }

        let last = vals[len - 1];
        let monotone = !(last < 0.0) && last == last;

        let mut flag = vec![0xDEADu32; len];
        let mut excl = 0.0f32;
        let mut t = 0usize;
        while t + UNROLL <= len {
            for k in 0..UNROLL {
                flag[t + k] = u32::from(excl < cut);
                excl += vals[t + k];
            }
            t += UNROLL;
            if !(excl < cut) && monotone {
                break;
            }
        }
        while t < len {
            flag[t] = u32::from(excl < cut);
            excl += vals[t];
            t += 1;
            if !(excl < cut) && monotone {
                break;
            }
        }
        let stop = t;

        for i in 0..len {
            keep[rowb + i] = pos[i] < stop && flag[pos[i]] != 0;
        }
    }
    keep
}

#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn host_top_p(values: &[f32], len: usize, cut: f32) -> Vec<bool> {
    let rows = values.len() / len;
    let mut keep = vec![false; values.len()];
    for row in 0..rows {
        let rowb = row * len;
        let slice = &values[rowb..rowb + len];
        let mut order: Vec<u32> = (0..len as u32).collect();
        order.sort_by(|&a, &b| desc_by_value(slice, a, b));
        let mut excl = 0.0f32;
        for &idx in &order {
            keep[rowb + idx as usize] = excl < cut;
            excl += slice[idx as usize];
        }
    }
    keep
}

fn the_top_p_walk_is_the_hosts_walk() {
    for &len in &[1usize, 2, 3, 5, 8, 17, 33, 64] {
        for &rows in &[1usize, 3] {
            for shape in 0..6u32 {
                let n = rows * len;
                let values: Vec<f32> = (0..n)
                    .map(|i| {
                        let t = (i % 11) as f32;
                        match shape {
                            0 => t * 0.1,
                            1 => t * 0.1 - 0.3,
                            2 => -t * 0.1 - 0.05,
                            3 => {
                                if i % 7 == 2 {
                                    f32::NAN
                                } else {
                                    t * 0.1 - 0.2
                                }
                            }
                            4 => match i % 5 {
                                0 => 0.0,
                                1 => -0.0,
                                2 => f32::INFINITY,
                                3 => f32::NEG_INFINITY,
                                _ => t * 0.1,
                            },
                            _ => ((i % 3) as f32) * 0.25,
                        }
                    })
                    .collect();

                for &cut in &[
                    f32::NEG_INFINITY,
                    -1.0,
                    0.0,
                    0.05,
                    0.5,
                    1.0,
                    1e30,
                    f32::INFINITY,
                ] {
                    assert_eq!(
                        top_p_walk(&values, len, cut),
                        host_top_p(&values, len, cut),
                        "shape {shape} at len {len} rows {rows} cut {cut} \
                         disagrees with the host's ordered walk"
                    );
                }
            }
        }
    }
}

#[allow(clippy::neg_cmp_op_on_partial_ord, clippy::eq_op)]
fn the_monotone_guard_is_the_no_negative_lane_predicate() {
    for x in [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1e-30,
        -1e-30,
    ] {
        assert_eq!(
            !(x < 0.0) && x == x,
            !(x < 0.0 || x != x),
            "{x} is on the wrong side of the walk's monotone guard"
        );
        assert_eq!(
            sort_key(x) > sort_key(0.0),
            x < 0.0 || x != x,
            "{x} is on the wrong side of the sorted row's negative suffix"
        );
    }
}
