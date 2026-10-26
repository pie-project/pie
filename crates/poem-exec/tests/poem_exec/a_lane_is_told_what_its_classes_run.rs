use poem_compiler::{Budget, Budgets, DeviceProfile, VoxelLadder, compile_axes};
use poem_ir::{LaneClasses, RowAxis, Trace, Ty};

/// What a lane of `switches` reads, over every value of the runtime switches.
fn lane(
    trace: &Trace,
    classes: &LaneClasses,
    switches: &[(&str, &str)],
) -> (bool, bool, Vec<String>) {
    let landed: Vec<_> = trace
        .facts
        .requests()
        .into_iter()
        .filter(|request| request.switches().is_empty())
        .map(|request| {
            switches.iter().fold(request, |request, (name, value)| {
                request.switched(name, value)
            })
        })
        .filter_map(|request| classes.of(trace.facts.word(&request)))
        .collect();
    let mut outputs: Vec<String> = landed
        .iter()
        .flat_map(|class| class.outputs.iter().map(|decl| decl.name.clone()))
        .collect();
    outputs.sort();
    outputs.dedup();
    (
        landed.iter().any(|class| class.reads_kv),
        landed.iter().any(|class| class.embeds),
        outputs,
    )
}

fn classes_of(name: &str) -> (Trace, LaneClasses) {
    let trace = poem_compiler::catalog::deployment(name)
        .unwrap_or_else(|| panic!("the catalog ships `{name}`"))
        .trace(poem_ir::Platform::Cuda);
    let voxels = trace.values.iter().any(|decl| {
        matches!(&decl.ty, Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(RowAxis::Voxels))
    });
    let mut budgets = Budgets::of(Budget::new(4, 1024));
    if voxels {
        budgets = budgets.with_voxels(VoxelLadder::new(4096, 4));
    }
    let compiled = compile_axes(&trace, &budgets, &DeviceProfile::default())
        .unwrap_or_else(|why| panic!("`{name}` bakes: {why}"));
    let outputs = poem_exec::feeds::class_outputs(&trace, &compiled)
        .unwrap_or_else(|why| panic!("`{name}` reads out: {why}"));
    let classes = poem_exec::feeds::lane_classes(&trace, &compiled, &outputs);
    (trace, classes)
}

type Told<'a> = (&'a [(&'a str, &'a str)], (bool, bool, &'a [&'a str]));

#[test]
fn a_lane_is_told_what_its_classes_run_every_case() {
    let rows: [(&str, &[Told]); 6] = [
        (
            "flux2-klein-4b-bf16-kv-bf16",
            &[
                (&[("reading", "text")], (true, true, &["hidden"])),
                (
                    &[("reading", "denoise"), ("stream", "image")],
                    (false, false, &["velocity"]),
                ),
                (
                    &[("reading", "denoise"), ("stream", "text")],
                    (false, false, &[]),
                ),
                (&[("reading", "vae.decode")], (false, false, &["pixels"])),
                (&[("reading", "vae.encode")], (false, false, &["pixels"])),
            ],
        ),
        (
            "z-image-mini-bf16-kv-bf16",
            &[
                (&[("reading", "refine")], (false, false, &["hidden"])),
                (
                    &[("reading", "denoise"), ("stream", "image")],
                    (false, false, &["velocity"]),
                ),
            ],
        ),
        (
            "flux2-mini-bf16-kv-bf16",
            &[
                (
                    &[("reading", "denoise"), ("stream", "image")],
                    (false, false, &["velocity"]),
                ),
                (
                    &[("reading", "denoise"), ("stream", "text")],
                    (false, false, &[]),
                ),
            ],
        ),
        (
            "wan22-mini-nano-bf16-kv-bf16",
            &[
                (
                    &[("reading", "denoise"), ("stream", "video")],
                    (false, false, &["velocity"]),
                ),
                (
                    &[("reading", "denoise"), ("stream", "context")],
                    (false, false, &[]),
                ),
            ],
        ),
        (
            "mini-dit-bf16-kv-bf16",
            &[(&[("stream", "image")], (false, false, &["velocity"]))],
        ),
        (
            "hunyuanimage3-mini-bf16-kv-bf16",
            &[
                (&[("reading", "text")], (true, true, &["hidden", "logits"])),
                (
                    &[("reading", "denoise")],
                    (true, true, &["hidden", "logits"]),
                ),
                (&[("reading", "image.out")], (false, false, &["pixels"])),
            ],
        ),
    ];
    for (name, lanes) in rows {
        let (trace, classes) = classes_of(name);
        for (switches, want) in lanes {
            let (kv, embeds, outputs) = lane(&trace, &classes, switches);
            assert_eq!(
                (
                    kv,
                    embeds,
                    outputs.iter().map(String::as_str).collect::<Vec<_>>()
                ),
                (want.0, want.1, want.2.to_vec()),
                "`{name}`, a lane of {switches:?}"
            );
        }
    }
}
