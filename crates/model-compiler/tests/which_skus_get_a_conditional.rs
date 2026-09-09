use model_compiler::{
    CompiledModel, DeviceProfile, Lowering, Phase, Region, collectives_are_never_elided,
};

mod common;
use common::{PLATFORMS, bake_with};
use model_ir::Trace;

fn forced() -> DeviceProfile {
    DeviceProfile {
        fat_region_us: 0.0,
        cond_fixed_us: 0.5,
        cond_per_arm_us: 0.0,
        ..DeviceProfile::default()
    }
}

fn cost(trace: &Trace, region: &Region, profile: &DeviceProfile) -> f32 {
    region
        .nodes
        .clone()
        .filter_map(|node| trace.nodes.get(node as usize))
        .map(|node| profile.family_us.of(&node.op))
        .sum()
}

#[allow(clippy::cast_precision_loss)]
fn launches(region: &Region) -> f32 {
    region.nodes.len() as f32
}

fn admits(trace: &Trace, compiled: &CompiledModel, at: usize, profile: &DeviceProfile) -> bool {
    let region = &compiled.regions[at];
    let windowed = region.phase == Phase::Capture
        && !region.collective
        && !region.mask.is_empty()
        && region.mask.len() < compiled.classes.classes.len();
    let arms = match region.lowering {
        Lowering::Switch { arms, .. } => arms,
        _ => 1,
    };
    let paid = profile.cond_fixed_us + profile.cond_per_arm_us * f32::from(arms);
    windowed
        && cost(trace, region, profile) >= profile.fat_region_us
        && launches(region) * profile.empty_launch_us > paid
}

#[test]
fn every_conditional_region_clears_every_gate_and_every_other_one_does_not() {
    let mut wrong: Vec<String> = Vec::new();

    for row in models::skus() {
        let (sku, trace) = (row.name.as_str(), row.trace);
        for platform in PLATFORMS {
            let trace = trace(platform);
            for profile in [DeviceProfile::default(), forced()] {
                let Ok(compiled) = bake_with(&trace, &profile) else {
                    continue;
                };
                for at in 0..compiled.regions.len() {
                    let region = &compiled.regions[at];
                    let conditional = region.lowering != Lowering::AlwaysLaunch;
                    if conditional != admits(&trace, &compiled, at, &profile) {
                        wrong.push(format!(
                            "`{sku}` as {platform:?} at fat={}: region {at} lowered \
                             {:?} and the gates say {}",
                            profile.fat_region_us,
                            region.lowering,
                            admits(&trace, &compiled, at, &profile),
                        ));
                    }
                    if conditional && (region.stream != 0 || region.open.is_some()) {
                        wrong.push(format!(
                            "`{sku}` as {platform:?}: region {at} is a conditional body \
                             on stream {} — a body is single-stream (design §4)",
                            region.stream,
                        ));
                    }
                    if conditional && region.close.is_some() {
                        wrong.push(format!(
                            "`{sku}` as {platform:?}: region {at} is a conditional body \
                             that closes a fork group",
                        ));
                    }
                }
                if !collectives_are_never_elided(&compiled) {
                    wrong.push(format!(
                        "`{sku}` as {platform:?}: a collective region became a \
                         conditional body — decision #5",
                    ));
                }
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}
