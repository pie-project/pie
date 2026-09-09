use eta_ir::op::Op;
use eta_ir::registry::{KernelInfo, ModelProfile};
use eta_ir::validate::bind;

use eta_dsl::builder::Builder;
use eta_dsl::prelude::*;
use eta_dsl::{Channel, Traced};

const V: u32 = 8;
const PAGES: u32 = 4;
const PAGE_T: u32 = 2;

fn quest_profile(with_kernel: bool) -> ModelProfile {
    let mut p = ModelProfile {
        vocab: V,
        page_size: PAGE_T,
        num_layers: 2,
        ..ModelProfile::dummy()
    };
    if with_kernel {
        p.kernels.push(KernelInfo {
            name: "envelope_dot".into(),
            sink_scope: None,
            replayable: true,
        });
    }
    p
}

fn quest_tap() -> Traced {
    let acc = Channel::from(vec![f32::NEG_INFINITY; PAGES as usize]).named("quest_acc");
    let out = Channel::new([PAGES], eta_dsl::dtype::f32).named("quest_scores");
    let acc_tap = acc.clone();
    let acc_epi = acc.clone();
    let mut b = Builder::new(V, PAGE_T);
    b.stage(Stage::OnAttnProj, move || {
        let prev = acc_tap.take();
        let scores = intrinsics::kernel::envelope_dot(PAGES);
        acc_tap.put(max_elem(&prev, &scores));
    });
    b.stage(Stage::Epilogue, move || {
        out.put(acc_epi.take());
        acc_epi.put(broadcast(f32::NEG_INFINITY, [PAGES]));
    });
    b.build().expect("the quest tap traces")
}

#[test]
fn kernel_call_every_case() {
    repeated_kernel_names_intern_once();
    the_tap_is_refused_without_the_kernel_in_the_profile();
    the_name_table_is_sorted_and_indices_are_remapped();
}

fn repeated_kernel_names_intern_once() {
    let sink = Channel::new([PAGES], eta_dsl::dtype::f32).named("sink");
    let mut b = Builder::new(V, PAGE_T);
    b.stage(Stage::OnAttnProj, || {
        let a = intrinsics::kernel::envelope_dot(PAGES);
        let c = intrinsics::kernel::envelope_dot(PAGES);
        let _ = max_elem(&a, &c);
    });
    b.stage(Stage::Epilogue, move || {
        sink.put(intrinsics::kernel::envelope_dot(PAGES));
    });
    let t = b.build().expect("traces");
    assert_eq!(t.container().names, vec!["envelope_dot".to_string()]);
    let calls: Vec<u16> = t
        .container()
        .stages
        .iter()
        .flat_map(|s| s.ops.iter())
        .filter_map(|op| match op {
            Op::KernelCall { name, .. } => Some(*name),
            _ => None,
        })
        .collect();
    assert_eq!(calls, vec![0, 0, 0], "one name, one index");
}

fn the_tap_is_refused_without_the_kernel_in_the_profile() {
    let t = quest_tap();
    let err = bind(t.container().clone(), quest_profile(false)).expect_err("must not bind");
    let msg = format!("{err:?}").to_lowercase();
    assert!(
        msg.contains("envelope_dot") || msg.contains("kernel"),
        "error should name the missing kernel, got: {msg}"
    );
    bind(t.container().clone(), quest_profile(true)).expect("binds once the profile advertises it");
}

fn the_name_table_is_sorted_and_indices_are_remapped() {
    let mut b = Builder::new(V, PAGE_T);
    b.stage(Stage::OnAttnProj, || {
        let scores = intrinsics::kernel::envelope_dot(PAGES);
        intrinsics::kernel::attn_page_mask(gt(&scores, 0.0f32));
    });
    let t = b.build().expect("two second-party names trace");
    let c = t.container();

    assert_eq!(
        c.names,
        vec!["attn_page_mask".to_string(), "envelope_dot".to_string()],
        "the name table must be sorted, not in first-use order"
    );
    let mut sorted = c.names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        c.names, sorted,
        "the table must be strictly sorted and unique"
    );

    let tap = c
        .stages
        .iter()
        .find(|s| s.stage == Stage::OnAttnProj)
        .expect("OnAttnProj stage");
    let kernel = tap
        .ops
        .iter()
        .find_map(|op| match op {
            Op::KernelCall { name, .. } => Some(*name),
            _ => None,
        })
        .expect("envelope_dot lowers to Op::KernelCall");
    let sink = tap
        .ops
        .iter()
        .find_map(|op| match op {
            Op::SinkCall { name, .. } => Some(*name),
            _ => None,
        })
        .expect("attn_page_mask lowers to Op::SinkCall");

    assert_eq!(
        c.names[kernel as usize], "envelope_dot",
        "the kernel call must still resolve to envelope_dot after the remap"
    );
    assert_eq!(
        c.names[sink as usize], "attn_page_mask",
        "the sink call must still resolve to attn_page_mask after the remap"
    );
}
