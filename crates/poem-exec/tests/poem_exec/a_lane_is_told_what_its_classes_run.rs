//! What a lane of a model's switches binds and reads out comes from the
//! classes its words land in: this is what each generative family's
//! readings used to state by hand.

use poem_compiler::{Budget, Budgets, DeviceProfile, VoxelLadder, compile_axes};
use poem_ir::{LaneClasses, ReadoutKind, RowAxis, Trace, Ty};

/// What a lane of `switches` reads: kv, tokens, and its readout's kind,
/// over every value of the runtime switches.
fn lane(
    trace: &Trace,
    classes: &LaneClasses,
    switches: &[(&str, &str)],
) -> (bool, bool, Option<ReadoutKind>) {
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
    let readouts = || {
        landed
            .iter()
            .filter_map(|class| class.readout)
            .map(|r| r.kind)
    };
    (
        landed.iter().any(|class| class.reads_kv),
        landed.iter().any(|class| class.embeds),
        readouts()
            .find(|kind| *kind != ReadoutKind::Logits)
            .or_else(|| readouts().next()),
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
    let classes = poem_exec::feeds::lane_classes(&trace, &compiled);
    (trace, classes)
}

/// A lane of these switches, and whether it reads kv, embeds tokens, and
/// what it reads out.
type Told<'a> = (&'a [(&'a str, &'a str)], (bool, bool, Option<ReadoutKind>));

#[test]
fn a_lane_is_told_what_its_classes_run_every_case() {
    use ReadoutKind::{Hidden, Logits, Pixels, Velocity};
    let rows: [(&str, &[Told]); 6] = [
        (
            "flux2-klein-4b-bf16-kv-bf16",
            &[
                (&[("reading", "text")], (true, true, Some(Hidden))),
                (
                    &[("reading", "denoise"), ("stream", "image")],
                    (false, false, Some(Velocity)),
                ),
                (
                    &[("reading", "denoise"), ("stream", "text")],
                    (false, false, None),
                ),
                (&[("reading", "vae.decode")], (false, false, Some(Pixels))),
                (&[("reading", "vae.encode")], (false, false, Some(Pixels))),
            ],
        ),
        (
            "z-image-mini-bf16-kv-bf16",
            &[
                (&[("reading", "refine")], (false, false, Some(Hidden))),
                (
                    &[("reading", "denoise"), ("stream", "image")],
                    (false, false, Some(Velocity)),
                ),
            ],
        ),
        (
            "flux2-mini-bf16-kv-bf16",
            &[
                (
                    &[("reading", "denoise"), ("stream", "image")],
                    (false, false, Some(Velocity)),
                ),
                (
                    &[("reading", "denoise"), ("stream", "text")],
                    (false, false, None),
                ),
            ],
        ),
        (
            "wan22-mini-nano-bf16-kv-bf16",
            &[
                (
                    &[("reading", "denoise"), ("stream", "video")],
                    (false, false, Some(Velocity)),
                ),
                (
                    &[("reading", "denoise"), ("stream", "context")],
                    (false, false, None),
                ),
            ],
        ),
        (
            "mini-dit-bf16-kv-bf16",
            &[(&[("stream", "image")], (false, false, Some(Velocity)))],
        ),
        (
            "hunyuanimage3-mini-bf16-kv-bf16",
            &[
                (&[("reading", "text")], (true, true, Some(Logits))),
                (&[("reading", "denoise")], (true, true, Some(Hidden))),
                (&[("reading", "image.out")], (false, false, Some(Pixels))),
            ],
        ),
    ];
    for (name, lanes) in rows {
        let (trace, classes) = classes_of(name);
        for (switches, want) in lanes {
            assert_eq!(
                lane(&trace, &classes, switches),
                *want,
                "`{name}`, a lane of {switches:?}"
            );
        }
    }
}
