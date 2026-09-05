//! The WGSL arm's own gate: what it emits must survive the compiler both
//! portable shells put it through.
//!
//! `engine-wgpu` hands the emitted source to `wgpu`, which parses WGSL with
//! naga; `engine-vulkan` compiles the same source to SPIR-V with naga's
//! `spv-out`. So parsing, validating and lowering here is not a proxy for what
//! the shells do — it is the same front end and the same back end, and a
//! shader that fails here is one neither backend could have run.
//!
//! What this does not check is what the ops COMPUTE. That is the device
//! differential against `eta_exec`'s interpreter, which needs an adapter and
//! lives in the shells' own suites.

use eta_compiler::codegen::launch::{LaunchOp, LaunchPlanValue, LaunchStagePlan};
use eta_compiler::codegen::wgsl::{RUNTIME, WORKGROUP, emit_launch_steps};
use eta_compiler::plan::Dimension;
use eta_ir::op::tags;
use eta_ir::types::Dtype;

/// Compiles one WGSL source the way a shell would, and answers the SPIR-V it
/// would hand Vulkan.
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

/// A region body of the shape the emitter writes: calls in plan order with a
/// barrier between them, and a reduce spelled as its ladder.
fn body(calls: &str) -> String {
    format!(
        "{RUNTIME}\n@compute @workgroup_size({WORKGROUP})\n\
         fn guest_pass(@builtin(local_invocation_id) lid : vec3<u32>) {{\n  \
         tid = lid.x;\n  lanes = {WORKGROUP}u;\n{calls}}}\n"
    )
}

/// The runtime alone, with an entry that touches nothing, is the floor: if
/// this fails every emitted shader fails, whatever its body.
#[test]
fn the_runtime_compiles() {
    let words = compile(&body(""));
    assert_eq!(words[0], 0x0723_0203, "the first word is the SPIR-V magic");
    assert!(
        words.len() > 1000,
        "a runtime this size lowers to more than {} words",
        words.len()
    );
}

/// Every op the emitter claims, driven through `ptir_step`. The switch is
/// reached with a runtime parameter, so one call exercises the arms' shapes
/// even though only one runs.
#[test]
fn a_body_of_ordinary_ops_compiles() {
    let calls = (0..8)
        .map(|node| format!("  ptir_step({node}u);\n  storageBarrier();\n"))
        .collect::<String>();
    compile(&body(&calls));
}

/// The reduce ladder: seven levels and a finish, with the barriers the emitter
/// puts between them. A barrier reached by only some invocations is undefined,
/// and naga rejects one it cannot prove uniform — so this compiling is the
/// evidence that the ladder's shape is sound.
#[test]
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

/// The sort ladder: a seed, `SORT_ROUNDS` merge rounds, the clear, and the
/// finishing pass, with the barriers the emitter puts between them. The rounds
/// carry a barrier for the same reason the reduce's levels do — the round
/// after reads what the round before wrote — and a merge round is a heavier
/// body than a reduce level, so its shape is worth compiling on its own.
#[test]
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

/// `sort_key`, transcribed from the runtime, and the host's own comparator.
///
/// The device orders a row by sorting `(sort_key(value), index)` ascending
/// rather than by running a comparator, so the whole of "the device agrees
/// with `sort_desc_order`" rests on that encoding being order-isomorphic to
/// `eta_exec::op::desc_by_value`. That is the one step here that is reasoning
/// rather than transcription, so it is worth a test that does not need a
/// device — and it is a MIRROR of the WGSL, so it has to be edited whenever
/// `sort_key` is.
fn sort_key(x: f32) -> u32 {
    if x.is_nan() {
        return 0xFFFF_FFFF;
    }
    // `-0.0` and `+0.0` are equal to the host and must share a key.
    let b = if x == 0.0 { 0u32 } else { x.to_bits() };
    let asc = if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    };
    0xFFFF_FFFF - asc
}

/// `eta_exec::op::desc_by_value`: NaN after every number, ties by index.
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

/// The key order and the host's comparator are the same total order.
///
/// Every awkward float is in the row on purpose: both zeroes, because their
/// bit patterns disagree where the host does not; both infinities, because the
/// key must span them without reaching the NaN key; two NaNs, because they
/// order after everything and among themselves by index; and duplicates,
/// because a tie is what the index half of the key exists to break.
#[test]
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

    // The NaN key is the ceiling and no number may reach it, or a number would
    // order among the NaNs.
    for &x in &row {
        if !x.is_nan() {
            assert!(
                sort_key(x) < 0xFFFF_FFFF,
                "{x} takes the key NaN is meant to own"
            );
        }
    }
}

/// `op_sort_round`'s index arithmetic, transcribed from the runtime.
///
/// The ladder's correctness is block/run arithmetic and a binary search, none
/// of which the compile gate can see and none of which a device is needed to
/// check. Like `sort_key` above this is a MIRROR of the WGSL and has to be
/// edited with it; what it buys is that an off-by-one in `mid`, `end` or the
/// rank fails here rather than in a shell's differential.
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

/// The ladder sorts each row into the host's order, at lengths that are not
/// powers of two and with rows that must not bleed into each other.
///
/// An even round count is what lands the answer in buffer 0; the `swap` here
/// stands in for the ping-pong, so a round count of the wrong parity would
/// show up as a wrong answer rather than as a silent read of the other buffer.
#[test]
fn the_merge_ladder_orders_every_row() {
    // Lengths either side of a power of two, plus the degenerate ones.
    for &len in &[1usize, 2, 3, 5, 8, 9, 17, 32, 33, 64] {
        for &rows in &[1usize, 3] {
            let n = rows * len;
            // A deterministic spread with duplicates, both zeroes and a NaN
            // wherever the row is long enough to hold one.
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

/// The bindings a shell has to provide, in the order it binds them. Reading
/// them back off the emitted module is how the two stay in step: a binding
/// added here without one added there is a shader the shell cannot dispatch.
#[test]
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

/// **THE SHAPE THAT USES MORE THAN ONE WORKGROUP MUST ALSO COMPILE.**
///
/// A stepwise module is not one entry point but dozens, sharing the runtime,
/// each taking `global_invocation_id` and `num_workgroups`. naga validates a
/// module whole, so a module with this many entry points is a different thing
/// to lower than a module with one — and every step's pipeline is created from
/// this one module, so a single bad entry point costs the whole pass.
#[test]
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

    // A vocabulary-sized row: iota 1, exp 1, the reduce's 7 levels + finish,
    // and the sort's seed + 16 rounds + pre + step + pack. Sixteen rounds
    // rather than the runtime's 28 because `sort_rounds` bounds the ladder by
    // the row.
    assert_eq!(stepwise.steps.len(), 30, "every rung is its own dispatch");

    let words = compile(&stepwise.source);
    assert_eq!(words[0], 0x0723_0203, "the first word is the SPIR-V magic");

    // Every step names an entry point the module actually declares — a shell
    // creates a pipeline per step by this name, and a name with no entry point
    // is a pipeline that fails to create at registration.
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

/// `op_pivot_finish`'s top-`p` walk and `op_pivot_pack`, transcribed from the
/// runtime: the walk writes a flag per SORTED POSITION and a stop index, and
/// the pack turns those back into a keep per ELEMENT through the inverse of
/// the ladder's order.
///
/// Like `sort_key` and `merge_ladder` above this is a MIRROR of the WGSL and
/// has to be edited with it. What it buys is the two things about this arm
/// that are reasoning rather than transcription: **the early break**, which is
/// a claim about every lane it has NOT looked at, and **the stop index**,
/// which is a claim that the positions it never wrote are all unkept.
// `!(x < 0.0)` and `x == x` are not idioms here, they are TRANSCRIPTIONS: the
// shader spells the guard that way, and a mirror that spells it `is_nan` or
// `partial_cmp` stops being evidence about the shader. Both forms are pinned
// against each other in `the_monotone_guard_is_the_no_negative_lane_predicate`.
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

        // `op_sort_pre`: the values in sorted order, and the inverse order.
        let vals: Vec<f32> = order.iter().map(|&i| slice[i as usize]).collect();
        let mut pos = vec![0usize; len];
        for (t, &i) in order.iter().enumerate() {
            pos[i as usize] = t;
        }

        // The row's smallest value answers "no lane is negative", because the
        // order is descending and a NaN sorts last.
        let last = vals[len - 1];
        let monotone = !(last < 0.0) && last == last;

        // The walk. Flags are only defined below `stop`; the unrolled bulk can
        // only leave on a block boundary, which makes `stop` coarser but never
        // wrong, since a flag is written from the live accumulator either way.
        // NOT zeroed, because the shader does not zero it either: the flags
        // live on scratch the ladder was using, so a position the walk never
        // reached holds a leftover key. `stop` is the only thing that keeps
        // the pack away from it, so the mirror poisons the array to make that
        // load-bearing here too.
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

        // `op_pivot_pack`.
        for i in 0..len {
            keep[rowb + i] = pos[i] < stop && flag[pos[i]] != 0;
        }
    }
    keep
}

/// `eta_exec::op::tags::PIVOT_THRESHOLD`'s top-`p` fallback: the whole ordered
/// row, no break at all. This is the answer the device has to reproduce.
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

/// **THE BREAK IS SOUND, INCLUDING ON THE ROWS THAT MADE THE OLD ONE WRONG.**
///
/// The walk may stop as soon as the mass clears the cut only if the
/// accumulator can never fall back under it, and that is a property of the
/// ROW, not of the element just added: descending order puts the negatives
/// last, so a non-negative element says nothing about the ones after it. The
/// arm used to test the element (`!(v < 0.0)`), which is wrong in both
/// directions — it broke early on a mixed row where the host kept walking, and
/// on a row with a negative tail it never fired at all.
///
/// So the rows here are chosen to hold negatives in the tail, NaNs, both
/// zeroes and both infinities, and the cuts to fall before, inside and after
/// the nucleus — including a cut no prefix reaches, which is the row that has
/// to be walked to its end.
#[test]
fn the_top_p_walk_is_the_hosts_walk() {
    for &len in &[1usize, 2, 3, 5, 8, 17, 33, 64] {
        for &rows in &[1usize, 3] {
            for shape in 0..6u32 {
                let n = rows * len;
                let values: Vec<f32> = (0..n)
                    .map(|i| {
                        let t = (i % 11) as f32;
                        match shape {
                            // All non-negative: the monotone row, the one the
                            // break exists for.
                            0 => t * 0.1,
                            // A negative tail: the row that used to spin.
                            1 => t * 0.1 - 0.3,
                            // Negatives only.
                            2 => -t * 0.1 - 0.05,
                            // A NaN, which sorts last and poisons the sum.
                            3 => {
                                if i % 7 == 2 {
                                    f32::NAN
                                } else {
                                    t * 0.1 - 0.2
                                }
                            }
                            // Both zeroes and both infinities.
                            4 => match i % 5 {
                                0 => 0.0,
                                1 => -0.0,
                                2 => f32::INFINITY,
                                3 => f32::NEG_INFINITY,
                                _ => t * 0.1,
                            },
                            // Duplicates, so ties break by index.
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

/// The break's guard is "no lane in the row is negative or a NaN", and the
/// walk reads it off the row's LAST value rather than searching, because the
/// order is descending and a NaN sorts last. So `!(last < 0.0) && last == last`
/// has to be that predicate exactly. `-0.0` is the case worth pinning: it is
/// not negative however its bits read.
#[test]
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
        // And the guard agrees with where the key order puts the suffix.
        assert_eq!(
            sort_key(x) > sort_key(0.0),
            x < 0.0 || x != x,
            "{x} is on the wrong side of the sorted row's negative suffix"
        );
    }
}
