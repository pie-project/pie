use std::collections::BTreeSet;

use crate::mini_dit_dims;

use mini_dit_dims as model;
use poem::{Def, Dim, Platform, RuntimeInput, Trace, Ty, seam};

const ROW: &str = "mini-dit-bf16-kv-bf16";

/// The deployment `ROW`, tapped at `tap` where given (`-tap.{key}`).
fn row(tap: Option<&str>) -> poem_compiler::catalog::Deployment {
    let name = match tap {
        None => ROW.to_string(),
        Some(key) => format!("{ROW}-tap.{key}"),
    };
    poem_compiler::catalog::repository()
        .parse(&name)
        .unwrap_or_else(|| panic!("{name} names a mini-dit deployment"))
}

fn traced(tap: Option<&str>) -> Trace {
    row(tap).trace(Platform::Cuda)
}

fn velocity_width(plan: &Trace) -> u64 {
    let seams: Vec<_> = plan
        .seams
        .iter()
        .filter(|s| s.seam == seam::VELOCITY.name)
        .collect();
    assert_eq!(seams.len(), 1, "one velocity seam, found {seams:?}");
    let value = seams[0].values[0];
    match &plan.values[value.0 as usize].ty {
        Ty::Tensor { shape, .. } => match shape.as_slice() {
            [Dim::Tokens, Dim::Const(width)] => *width,
            other => panic!("the velocity seam is a `[Tokens, width]` rectangle, not {other:?}"),
        },
        other => panic!("the velocity seam is a tensor, not {other:?}"),
    }
}

fn ports(plan: &Trace) -> BTreeSet<&str> {
    plan.values
        .iter()
        .filter_map(|decl| match &decl.def {
            Def::Input(RuntimeInput::Guest { id, .. }) => {
                Some(plan.inputs[*id as usize].name.as_str())
            }
            _ => None,
        })
        .collect()
}

#[test]
fn the_mini_dit_tap_exports_an_intermediate_every_case() {
    a_tap_moves_the_velocity_seam_to_the_intermediate_at_its_width();
    a_tap_is_still_one_float_readout_and_no_logits();
    a_tap_that_cuts_the_trace_short_still_reads_every_port();
    an_unknown_key_is_refused();
    a_tap_is_named_in_the_deployment();
}

fn a_tap_moves_the_velocity_seam_to_the_intermediate_at_its_width() {
    assert_eq!(
        velocity_width(&traced(None)),
        u64::from(model::PATCH_FEATURES)
    );
    assert_eq!(
        velocity_width(&traced(Some("b0.norm1_out"))),
        u64::from(model::HIDDEN)
    );
    assert_eq!(
        velocity_width(&traced(Some("b2.out"))),
        u64::from(model::HIDDEN)
    );
    assert_eq!(
        velocity_width(&traced(Some("final.norm_out"))),
        u64::from(model::HIDDEN)
    );
}

fn a_tap_is_still_one_float_readout_and_no_logits() {
    for tap in ["x_embed", "b0.attn_heads", "b1.out_img", "b2.cross_q"] {
        let plan = traced(Some(tap));
        let seams: Vec<&str> = plan.seams.iter().map(|s| s.seam.as_str()).collect();
        assert_eq!(
            seams.iter().filter(|s| **s == seam::VELOCITY.name).count(),
            1,
            "{tap}: one velocity seam; seams are {seams:?}"
        );
        assert!(
            !seams.contains(&seam::OUT.name),
            "{tap}: a probe has no logits either; seams are {seams:?}"
        );
    }
}

fn a_tap_that_cuts_the_trace_short_still_reads_every_port() {
    let plan = traced(None);
    let whole = ports(&plan);
    assert!(
        whole.contains("context"),
        "the model reads the context port: {whole:?}"
    );
    for tap in ["x_embed", "b0.in", "b0.norm1_out", "b1.joint_attn_heads"] {
        assert_eq!(
            ports(&traced(Some(tap))),
            whole,
            "{tap}: the ports the plan declares"
        );
    }
}

fn an_unknown_key_is_refused() {
    let why = row(Some("not.a.dump.key"))
        .try_trace(Platform::Cuda)
        .expect_err("a tap that names no intermediate");
    assert!(
        format!("{why:?}").contains("names no intermediate"),
        "the refusal names the tap: {why:?}"
    );
}

fn a_tap_is_named_in_the_deployment() {
    assert_eq!(row(Some("b0.in")).name, format!("{ROW}-tap.b0.in"));
    assert_eq!(row(None).name, ROW);
}
