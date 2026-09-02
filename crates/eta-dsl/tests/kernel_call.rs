//! Emitting a `KernelCall`, via the `intrinsics::kernel::envelope_dot`
//! authoring surface.
//!
//! Until this lowering existed the DSL could not emit `Op::KernelCall` at all —
//! `builder.rs` hard-coded `names: Vec::new()` in both `TraceContainer`
//! constructions, so the name index a `KernelCall` refers to could never be
//! populated. These tests pin the two halves of the fix: names are interned
//! session-wide (a `NameIndex` is container-wide, not stage-wide), and the
//! emitted op survives `bind` against a profile that advertises the kernel.

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

/// Build the canonical Quest tap: per layer, envelope scores fold into a
/// device-carried `[PAGES]` accumulator; the epilogue publishes the fold.
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

/// Interning is session-scoped: two stages naming the same kernel share one
/// index, and the index is stable across the stage boundary.
#[test]
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

/// The tap must be refused against a backend that does not advertise the
/// kernel — this is the bind-time half of the CUDA `has_kv_envelopes` gate.
#[test]
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

/// The name table is emitted SORTED, and every `name_idx` follows it.
///
/// `intern_name` hands out indices in first-use order, but the container's
/// name table must be strictly sorted and unique — the loader rejects it
/// otherwise. Nothing caught this until a program used two second-party names
/// whose use order disagreed with their sort order, which is exactly what
/// Quest does once it both scores pages (`envelope_dot`) and acts on the score
/// (`attn_page_mask`): `attn_page_mask` is used second and sorts first.
///
/// The interesting half is the REMAP. Sorting the table while leaving the
/// indices alone would still load, and would silently invoke the wrong kernel.
#[test]
fn the_name_table_is_sorted_and_indices_are_remapped() {
    let mut b = Builder::new(V, PAGE_T);
    b.stage(Stage::OnAttnProj, || {
        // Used first, sorts second.
        let scores = intrinsics::kernel::envelope_dot(PAGES);
        // Used second, sorts first.
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
