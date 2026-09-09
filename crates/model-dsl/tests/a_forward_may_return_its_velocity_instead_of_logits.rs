use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, ops, seam,
    trace_hybrid,
};

struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

#[derive(Clone, Copy)]
enum Plants {
    Velocity,
    Hidden,
    Nothing,
    HiddenBesideLogits,
}

struct Readout(Plants);

impl ForwardHybrid for Readout {
    type Facts = NoFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let x = inputs.latents(0, 16, Dtype::Bf16);
        let w = Weight::sym("w", [16, 16], Dtype::Bf16);
        let layers = [(); 2];
        let mut h = x;
        for (_, ()) in inputs.walk_layers(&layers) {
            h = ops::linear::matmul(&h, &w);
            if matches!(self.0, Plants::HiddenBesideLogits) {
                seam::at(seam::HIDDEN, &[&h]);
            }
        }
        match self.0 {
            Plants::Velocity => seam::at(seam::VELOCITY, &[&h]),
            Plants::Hidden => seam::at(seam::HIDDEN, &[&h]),
            Plants::Nothing => {}
            Plants::HiddenBesideLogits => {
                let head = Weight::sym("head", [32, 16], Dtype::Bf16);
                h = ops::linear::lm_head(&h, &head);
            }
        }
        h
    }
}

fn seams(plants: Plants) -> Vec<(String, usize, Option<u32>)> {
    trace_hybrid("readout", &Readout(plants), Platform::Cuda)
        .seams
        .iter()
        .map(|seam| (seam.seam.clone(), seam.values.len(), seam.layer))
        .collect()
}

fn count(seams: &[(String, usize, Option<u32>)], name: &str) -> usize {
    seams.iter().filter(|(seam, _, _)| seam == name).count()
}

#[test]
fn a_forward_may_return_its_velocity_instead_of_logits_every_case() {
    velocity_on_the_returned_value_stands_in_for_out();
    hidden_on_the_returned_value_stands_in_for_out();
    nothing_planted_still_gets_out();
    hidden_beside_logits_keeps_both_and_names_its_layer();
}

fn velocity_on_the_returned_value_stands_in_for_out() {
    let seams = seams(Plants::Velocity);
    assert_eq!(count(&seams, seam::OUT.name), 0, "{seams:?}");
    assert_eq!(count(&seams, seam::VELOCITY.name), 1, "{seams:?}");
}

fn hidden_on_the_returned_value_stands_in_for_out() {
    let seams = seams(Plants::Hidden);
    assert_eq!(count(&seams, seam::OUT.name), 0, "{seams:?}");
    assert_eq!(count(&seams, seam::HIDDEN.name), 1, "{seams:?}");
}

fn nothing_planted_still_gets_out() {
    let seams = seams(Plants::Nothing);
    assert_eq!(count(&seams, seam::OUT.name), 1, "{seams:?}");
    assert_eq!(count(&seams, seam::VELOCITY.name), 0);
    assert_eq!(count(&seams, seam::HIDDEN.name), 0);
}

fn hidden_beside_logits_keeps_both_and_names_its_layer() {
    let seams = seams(Plants::HiddenBesideLogits);
    assert_eq!(count(&seams, seam::OUT.name), 1, "{seams:?}");
    let hidden: Vec<Option<u32>> = seams
        .iter()
        .filter(|(seam, _, _)| seam == seam::HIDDEN.name)
        .map(|(_, _, layer)| *layer)
        .collect();
    assert_eq!(
        hidden,
        vec![Some(0), Some(1)],
        "one hidden tap per layer, each naming it"
    );
    assert_eq!(
        seam::FLOAT_READOUTS,
        [seam::VELOCITY.name, seam::HIDDEN.name, seam::PIXELS.name]
    );
}
