//! The sizing pass (`Shell::size_outside`) compiles only the ladder's fires
//! of each part's widest rows, at every lane count, fewer than the ladder
//! lays; and every fire the ladder then lays past what it measured is named
//! (`Shell::past_outside`), each truly past it (a fire's temp moves with
//! XLA's memory-space assignment, not monotonically with its rows or
//! lanes). Compiled on the device, against XLA's memory analysis.

use std::path::Path;

use engine_xla::{Boot, DeviceBoot, Shell};
use poem::Platform;
use poem_compiler::Budget;

const PAGE: u32 = 16;
const CONTEXT: u32 = 256;
const SLOTS: u32 = 2;

#[test]
fn the_sizing_pass_measures_its_fires_and_names_every_fire_past_them() {
    let deployment = poem_compiler::catalog::deployment("qwen35-d0.8b-bf16-kv-bf16")
        .expect("the deployment is listed");
    let trace = deployment.trace(Platform::Xla);
    let paging =
        engine_xla::store::kv::Paging::of(PAGE, CONTEXT, SLOTS, u64::from(SLOTS * CONTEXT / PAGE))
            .expect("the paging lays out");
    let kv_budget = engine_xla::store::pool_demand(&trace, paging).expect("the pools size");
    let contract = checkpoint::contract::ModelContract {
        alignment: 1,
        tensors: Vec::new(),
        groups: Vec::new(),
    };
    let rules = engine::StepRules::runtime();
    let mut shell = Shell::load_dry(
        Boot {
            trace,
            contract: &contract,
            checkpoint: Path::new("/dev/null"),
            budget: Budget {
                max_lanes: 8,
                max_tokens: 128,
                buckets: Vec::new(),
                max_adapters: 0,
                readouts_per_lane: poem_compiler::arena::READOUTS_PER_LANE,
            },
            patches: None,
            page_size: PAGE,
            context: CONTEXT,
            slots: SLOTS,
            kv_budget: Some(kv_budget),
            allowance: engine::Allowance::WHOLE,
            device: &DeviceBoot::default(),
            steps: Some(&rules),
        },
        None,
        true,
    )
    .expect("the compiling dry shell loads");
    shell
        .warm_maintenance()
        .expect("the maintenance programs lay");
    let stated = shell.stated_steps(&rules);
    shell.set_steps(stated.shapes);
    let outside = shell.size_outside().expect("the sizing pass compiles");
    assert!(outside.programs > 0, "the sizing pass compiled no fire");
    let sized = shell.device().memories().len();
    shell.prewarm().expect("the ladder lays");
    let laid = shell.device().memories().len();
    assert!(
        laid > sized,
        "the ladder laid no fire past the sizing pass's {sized}"
    );
    let past = shell.past_outside(outside.bytes());
    // each fire past it is a program of some stated shape, stated unservable
    let unservable = shell.unservable_shapes(outside.bytes());
    assert_eq!(
        past.is_empty(),
        unservable.is_empty(),
        "fires past the measure {past:?}, shapes stated unservable {unservable:?}"
    );
    assert!(
        past.iter().all(|(_, bytes)| *bytes > outside.bytes()),
        "a fire named past the {} bytes measured is not: {past:?}",
        outside.bytes()
    );
    assert!(
        outside.programs < laid - sized,
        "the sizing pass compiled {} programs, the ladder {} past them",
        outside.programs,
        laid - sized
    );
    println!(
        "sizing pass: {} programs, {} MiB; ladder {} more; {} fires past it",
        outside.programs,
        outside.bytes() >> 20,
        laid - sized,
        past.len()
    );
}
