use crate::msl_corpus::{corpus_stages, extended_stages};
use crate::msl_mutations::{MUTATIONS, mutate};
use eta_compiler::codegen::metal::validate_singleton_plan;

const TOLERATED: &[(&str, &str)] = &[(
    "none",
    "the unmutated plan; here to prove the harness accepts something",
)];

fn verdicts() -> Vec<(&'static str, usize, usize)> {
    let stages: Vec<_> = corpus_stages()
        .into_iter()
        .chain(extended_stages())
        .filter(|stage| validate_singleton_plan(&stage.plan).is_ok())
        .collect();
    MUTATIONS
        .iter()
        .map(|mutation| {
            let (mut applied, mut rejected) = (0, 0);
            for stage in &stages {
                let mut damaged = stage.plan.clone();
                if !mutate(&mut damaged, mutation) {
                    continue;
                }
                applied += 1;
                if validate_singleton_plan(&damaged).is_err() {
                    rejected += 1;
                }
            }
            (*mutation, applied, rejected)
        })
        .collect()
}

#[test]
fn singleton_plan_rejection_is_total() {
    let mut holes = Vec::new();
    for (mutation, applied, rejected) in verdicts() {
        let tolerated = TOLERATED.iter().any(|(name, _)| *name == mutation);
        if tolerated {
            assert_eq!(
                rejected, 0,
                "{mutation:?} is listed in TOLERATED but is rejected on \
                 {rejected}/{applied} stages — it is a real check now, so take it \
                 out of the list"
            );
            continue;
        }
        if rejected != applied {
            holes.push(format!(
                "{mutation}: caught on {rejected} of {applied} stages"
            ));
        }
    }
    assert!(
        holes.is_empty(),
        "validate_singleton_plan lets damaged plans through:\n  {}\n\
         Either the check is missing, or the mutation is harmless and belongs \
         in TOLERATED with the reason why.",
        holes.join("\n  ")
    );
}
