//! The compiler's per-chip cost coefficients, as device-profile decisions.

use profile::{Decision, Profile, Registry};

use crate::budget::{DeviceProfile, FamilyCosts};

macro_rules! micros {
    ($($ty:ident = $id:literal, $default:literal, $doc:literal;)*) => {
        $(
            #[doc = $doc]
            #[derive(Clone, Copy, Debug, PartialEq, Decision)]
            #[decision(id = $id, constant(range(0..=1_000_000)), default = $default, tune = price)]
            pub struct $ty(pub f64);
        )*
    };
}

micros! {
    LaunchUs = "compiler.launch_us", 1.0, "An empty kernel launch, in microseconds.";
    CondFixedUs = "compiler.cond_fixed_us", 5.0, "A conditional node's fixed cost, in microseconds.";
    CondPerArmUs = "compiler.cond_per_arm_us", 0.6, "Each arm a conditional node adds, in microseconds.";
    FatRegionUs = "compiler.fat_region_us", 250.0, "A region long enough to leave unsplit, in microseconds.";
    ForkFloorUs = "compiler.fork_floor_us", 20.0, "The least work a side stream is forked for, in microseconds.";
    AttentionUs = "compiler.cost.attention_us", 60.0, "An attention op, in microseconds.";
    LinearUs = "compiler.cost.linear_us", 40.0, "A matmul, in microseconds.";
    ElementwiseUs = "compiler.cost.elementwise_us", 4.0, "An elementwise op, in microseconds.";
    LayoutUs = "compiler.cost.layout_us", 4.0, "A layout op, in microseconds.";
    CollectiveUs = "compiler.cost.collective_us", 50.0, "A collective, in microseconds.";
    CustomUs = "compiler.cost.custom_us", 20.0, "A custom fused op, in microseconds.";
    SpatialUs = "compiler.cost.spatial_us", 40.0, "A spatial op, in microseconds.";
}

/// Side streams the compiler may fork independent regions onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "compiler.side_streams", constant(range(0..=8)), default = 2, tune = engine)]
pub struct SideStreams(pub u32);

/// Register the cost decisions, with this backend's own side-stream default.
pub fn register(registry: &mut Registry, side_streams: u32) {
    registry
        .add::<LaunchUs>()
        .add::<CondFixedUs>()
        .add::<CondPerArmUs>()
        .add::<FatRegionUs>()
        .add::<ForkFloorUs>()
        .add::<AttentionUs>()
        .add::<LinearUs>()
        .add::<ElementwiseUs>()
        .add::<LayoutUs>()
        .add::<CollectiveUs>()
        .add::<CustomUs>()
        .add::<SpatialUs>()
        .add_defaulting::<SideStreams>(&side_streams.to_string());
}

impl DeviceProfile {
    /// The compiler's prices from a resolved profile, for a device with `sms`
    /// multiprocessors (or GPU cores). `exclusive` and `grouped` name the ops
    /// the backend serves alone or as one grouped launch.
    #[must_use]
    pub fn of(profile: &Profile, sms: u32, exclusive: &[&str], grouped: &[&str]) -> DeviceProfile {
        let us = |v: f64| v as f32;
        DeviceProfile {
            sms,
            empty_launch_us: us(profile.get::<LaunchUs>().0),
            cond_fixed_us: us(profile.get::<CondFixedUs>().0),
            cond_per_arm_us: us(profile.get::<CondPerArmUs>().0),
            fat_region_us: us(profile.get::<FatRegionUs>().0),
            side_streams: profile.get::<SideStreams>().0,
            fork_floor_us: us(profile.get::<ForkFloorUs>().0),
            family_us: FamilyCosts {
                attention: us(profile.get::<AttentionUs>().0),
                linear: us(profile.get::<LinearUs>().0),
                elementwise: us(profile.get::<ElementwiseUs>().0),
                layout: us(profile.get::<LayoutUs>().0),
                collective: us(profile.get::<CollectiveUs>().0),
                custom: us(profile.get::<CustomUs>().0),
                spatial: us(profile.get::<SpatialUs>().0),
            },
            exclusive: exclusive.iter().map(|s| (*s).to_string()).collect(),
            grouped: grouped.iter().map(|s| (*s).to_string()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_defaults_are_the_compilers_own() {
        let mut r = Registry::new("toy", &[], &["rows"]);
        register(&mut r, 2);
        let from = DeviceProfile::of(&r.defaults(), 132, &[], &[]);
        assert_eq!(from, DeviceProfile::default());
        let mut metal = Registry::new("toy", &[], &["rows"]);
        register(&mut metal, 0);
        assert_eq!(
            DeviceProfile::of(&metal.defaults(), 20, &[], &[]).side_streams,
            0
        );
    }
}
