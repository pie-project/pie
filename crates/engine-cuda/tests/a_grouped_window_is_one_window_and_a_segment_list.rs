use engine_cuda::window::{Copies, Windows};
use model_compiler::{Budget, CompiledModel, DeviceProfile, FamilyCosts, compile};
use model_dsl::Platform;
use model_exec::fire::{Lane, compose};

use model_ir::Trace;

fn test_slots() -> engine_cuda::window::Slots {
    engine_cuda::window::Slots::new(8, 512, 8, 1, 4096, 2, 512)
}

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

const CORRECTION: &str = "linear.lora_correct";

fn budget() -> Budget {
    Budget {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: 8,
    }
}

fn arms() -> (DeviceProfile, DeviceProfile) {
    let split = DeviceProfile {
        family_us: FamilyCosts {
            linear: 1.0,
            ..DeviceProfile::default().family_us
        },
        ..DeviceProfile::default()
    };
    let grouped = DeviceProfile {
        grouped: vec![CORRECTION.to_string()],
        ..DeviceProfile::default()
    };
    (split, grouped)
}

fn sku() -> (Trace, CompiledModel, CompiledModel) {
    let trace = models::sku(SKU)
        .unwrap_or_else(|| panic!("`{SKU}` is in the catalog"))
        .trace;
    let trace = trace(Platform::Cuda);
    let (split, grouped) = arms();
    let split = compile(&trace, &budget(), &split).expect("the split arm bakes");
    let grouped = compile(&trace, &budget(), &grouped).expect("the grouped arm bakes");
    (trace, split, grouped)
}

fn indptr(rows: &[u32]) -> Vec<i32> {
    let mut out = vec![0i32];
    for &n in rows {
        out.push(out[out.len() - 1] + n as i32);
    }
    out
}

fn one_lane_per_class(compiled: &CompiledModel) -> Vec<Lane> {
    compiled
        .classes
        .classes
        .iter()
        .map(|class| Lane::new(class.word(), if class.word() & 1 == 1 { 1 } else { 3 }))
        .collect()
}

#[test]
fn the_segment_lists_are_staged_beside_the_boundaries_in_the_one_copy() {
    let (plan, _, grouped) = sku();
    let lanes = one_lane_per_class(&grouped);
    let fire = compose(&grouped, &budget(), &lanes).expect("eight lanes compose");
    let rows: Vec<u32> = fire.lanes().iter().map(|lane| lane.rows).collect();
    let mut windows = Windows::of(
        &plan,
        &grouped,
        model_ir::PerAxis::new([fire.classes(), fire.patch_classes(), fire.voxel_classes()]),
        &indptr(&rows),
        Copies::off(),
        test_slots(),
    )
    .expect("the windows");

    let packed = windows.packed();
    const BASE: u64 = 0x1000;
    windows.bind(BASE);

    let mut segmented = 0usize;
    for at in 0..grouped.template().len() as u32 {
        for run in 0..windows.runs(at) {
            let held = windows.at(at, run);

            let start = ((held.indptr.ptr - BASE) / 4) as usize;
            assert_eq!(
                &packed[start..start + held.indptr_host.len()],
                held.indptr_host.as_slice(),
                "region {at} run {run}: the staged boundaries are not this window's",
            );
            assert_eq!(held.indptr.rows as usize, held.indptr_host.len());

            if held.segs() == 0 {
                assert_eq!(held.segments.rows, 0);
                continue;
            }
            segmented += 1;

            let after = start + held.indptr_host.len();
            assert_eq!(
                held.segments.ptr,
                BASE + (after as u64) * 4,
                "region {at}: the segments do not follow this window's boundaries",
            );
            assert_eq!(
                &packed[after..after + held.segments_host.len()],
                held.segments_host.as_slice(),
            );
            assert_eq!(held.segments.rows, held.segs());
            assert_eq!(held.segments.width, 2, "an `[segs, 2]` rectangle");
        }
    }
    assert!(
        segmented > 0,
        "no window in this fire carries a segment list, and then the staging \
         layout under test is never exercised",
    );
}
