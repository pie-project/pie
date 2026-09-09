#![allow(dead_code)]

use model_compiler::{
    Budget, Budgets, CompiledModel, DeviceProfile, Error, PATCH_LATTICE_FLOOR, PatchLadder,
    compile_axes,
};
use model_dsl::Platform;
use model_ir::{ParamSource, RowAxis, Trace, Ty};

pub const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

pub fn budgets_for(trace: &Trace) -> Budget {
    let seats = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Registered)
        .map(|param| param.shape.first().copied().unwrap_or(0))
        .min()
        .unwrap_or(0);
    Budget {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: u32::try_from(seats).unwrap_or(u32::MAX),
    }
}

pub fn states_patches(trace: &Trace) -> bool {
    trace.values.iter().any(|decl| {
        matches!(&decl.ty, Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(RowAxis::Patches))
    })
}

pub fn patch_ladder_for(budget: &Budget) -> PatchLadder {
    const DERIVED_PATCH_CEILING: u32 = 4096;

    let max_patches = budget
        .max_tokens
        .clamp(PATCH_LATTICE_FLOOR, DERIVED_PATCH_CEILING);
    let mut buckets = Vec::new();
    let mut rung = PATCH_LATTICE_FLOOR;
    while rung < max_patches {
        buckets.push(rung);
        rung = rung.saturating_mul(2);
    }
    buckets.push(max_patches);
    PatchLadder {
        max_images: (max_patches / PATCH_LATTICE_FLOOR).max(1),
        max_patches,
        buckets,
    }
}

pub fn budgets_of(trace: &Trace) -> Budgets {
    let budget = budgets_for(trace);
    let budgets = Budgets::of(budget.clone());
    if states_patches(trace) {
        budgets.with_patches(patch_ladder_for(&budget))
    } else {
        budgets
    }
}

pub fn bake(trace: &Trace) -> Result<CompiledModel, Error> {
    bake_with(trace, &DeviceProfile::default())
}

pub fn bake_with(trace: &Trace, profile: &DeviceProfile) -> Result<CompiledModel, Error> {
    compile_axes(trace, &budgets_of(trace), profile)
}
