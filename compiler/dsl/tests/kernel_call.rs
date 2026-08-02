//! Emitting a `KernelCall`, via the `intrinsics::kernel::envelope_dot`
//! authoring surface.
//!
//! Until this lowering existed the DSL could not emit `Op::KernelCall` at all —
//! `builder.rs` hard-coded `names: Vec::new()` in both `TraceContainer`
//! constructions, so the name index a `KernelCall` refers to could never be
//! populated. These tests pin the two halves of the fix: names are interned
//! session-wide (a `NameIndex` is container-wide, not stage-wide), and the
//! emitted op survives `bind` against a profile that advertises the kernel.

use pie_eval::interp::{Instance, KernelHost, PassInputs, Value};
use pie_ir::op::Op;
use pie_ir::registry::{KernelInfo, ModelProfile};
use pie_ir::validate::bind;

use pie_dsl::builder::Builder;
use pie_dsl::prelude::*;
use pie_dsl::{Channel, DType, Traced};

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

/// `score[p] = p + 1` — deterministic and strictly increasing, so the
/// top-`BUDGET` selection has a unique answer.
struct RampKernels {
    calls: usize,
}

impl KernelHost for RampKernels {
    fn kernel(
        &mut self,
        name: &str,
        _args: &[Value],
        result: pie_ir::types::ValueType,
    ) -> Result<Value, String> {
        if name != "envelope_dot" {
            return Err(format!("no such kernel: {name}"));
        }
        self.calls += 1;
        let n = result.shape.numel() as usize;
        Ok(Value::F32((0..n).map(|p| p as f32 + 1.0).collect()))
    }
}

/// Build the canonical Quest tap: per layer, envelope scores fold into a
/// device-carried `[PAGES]` accumulator; the epilogue publishes the fold.
fn quest_tap() -> Traced {
    let acc = Channel::from(vec![f32::NEG_INFINITY; PAGES as usize]).named("quest_acc");
    let out = Channel::new([PAGES], pie_dsl::dtype::f32).named("quest_scores");
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
fn envelope_dot_interns_its_name_and_emits_a_kernel_call() {
    let t = quest_tap();
    let c = t.container();

    assert_eq!(
        c.names,
        vec!["envelope_dot".to_string()],
        "the kernel name must land in the container-wide name index"
    );

    let tap = c
        .stages
        .iter()
        .find(|s| s.stage == Stage::OnAttnProj)
        .expect("OnAttnProj stage");
    let call = tap
        .ops
        .iter()
        .find_map(|op| match op {
            Op::KernelCall {
                name, shape, dtype, ..
            } => Some((*name, *shape, *dtype)),
            _ => None,
        })
        .expect("the tap must lower to Op::KernelCall");
    assert_eq!(call.0, 0, "first interned name is index 0");
    assert_eq!(call.1.numel(), u64::from(PAGES));
    assert_eq!(call.2, DType::F32);
}

/// Interning is session-scoped: two stages naming the same kernel share one
/// index, and the index is stable across the stage boundary.
#[test]
fn repeated_kernel_names_intern_once() {
    let sink = Channel::new([PAGES], pie_dsl::dtype::f32).named("sink");
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

/// End-to-end on the reference interpreter: the tap fires once per layer, the
/// fold is a max over layers, and the epilogue re-seeds the accumulator so the
/// next fire starts clean.
#[test]
fn the_tap_folds_over_layers_and_reseeds() {
    let t = quest_tap();
    let profile = quest_profile(true);
    let layers = profile.num_layers as usize;
    let bound = bind(t.container().clone(), profile).expect("bind");

    let mut kernels = RampKernels { calls: 0 };
    // Declaration order: quest_acc (gid 0), quest_scores (gid 1).
    let seeds = [(0u32, Value::F32(vec![f32::NEG_INFINITY; PAGES as usize]))];
    let mut inst = Instance::new(&bound, &seeds).unwrap();
    let scores_ch = 1u32;
    let want: Vec<f32> = (0..PAGES).map(|p| p as f32 + 1.0).collect();
    // The tap declares its query operand as a shape-erased `[1]` handle: the
    // backend kernel reads the live query row, not this buffer.
    let inputs = PassInputs {
        query: vec![Value::F32(vec![1.0]); layers],
        ..PassInputs::default()
    };

    for fire in 0..2 {
        let r = inst
            .step(&bound, &inputs, &mut kernels)
            .unwrap_or_else(|e| panic!("fire {fire}: {e:?}"));
        assert!(
            r.committed,
            "fire {fire} must commit, missed={:?}",
            r.missed
        );
        let published = inst
            .host_take(&bound, scores_ch)
            .unwrap_or_else(|e| panic!("fire {fire}: take scores: {e:?}"));
        let Value::F32(v) = published else {
            panic!("fire {fire}: scores must be f32");
        };
        assert_eq!(
            v, want,
            "fire {fire}: the fold must be the max over layers of the ramp"
        );
    }
    assert_eq!(
        kernels.calls,
        2 * layers,
        "the tap must fire exactly once per layer per fire"
    );
}

/// `kernel::lora` (§6.5): a pass-wide configuration sink authored in the
/// prologue. Three args — A, B, SITES; no scalar alpha (alpha/R is folded
/// into B's contents); SITES trace-known. Gated at bind on the backend's
/// `has_lora`, exactly as `attn_page_mask` gates on `has_attn_page_mask`.
#[test]
fn lora_traces_to_a_prologue_sink_call_and_gates_on_the_profile() {
    const L: u32 = 2; // num_layers
    const R: u32 = 2; // rank — trace-known: a different R is a different program
    const D: u32 = 4;

    let la = Channel::from_shaped([L, R, D], vec![0.0f32; (L * R * D) as usize]).named("lora_a");
    let lb = Channel::from_shaped([L, D, R], vec![0.0f32; (L * D * R) as usize]).named("lora_b");
    let mut b = Builder::new(V, PAGE_T);
    b.stage(Stage::Prologue, move || {
        // Placement is structure (a trace-known constant over the site
        // vocabulary); weights are contents (channel reads — peeks, so the
        // sink adds no edge to the decode chain).
        intrinsics::kernel::lora(la.read(), lb.read(), Tensor::constant([0u32, 1, 2]));
    });
    let t = b.build().expect("the lora prologue traces");
    let c = t.container();

    assert_eq!(
        c.names,
        vec!["lora".to_string()],
        "the sink name must land in the container-wide name index"
    );
    let prologue = c
        .stages
        .iter()
        .find(|s| s.stage == Stage::Prologue)
        .expect("Prologue stage");
    let (name, args) = prologue
        .ops
        .iter()
        .find_map(|op| match op {
            Op::SinkCall { name, args } => Some((*name, args.clone())),
            _ => None,
        })
        .expect("lora lowers to Op::SinkCall");
    assert_eq!(name, 0, "first interned name is index 0");
    assert_eq!(
        args.len(),
        3,
        "A, B, SITES — and nothing else: the scale is per-adapter data"
    );

    bind(t.container().clone(), ModelProfile::dummy())
        .expect("binds when the profile advertises has_lora");
    let mut no_lora = ModelProfile::dummy();
    no_lora.has_lora = false;
    let err = bind(t.container().clone(), no_lora).expect_err("must be refused without has_lora");
    let msg = format!("{err:?}").to_lowercase();
    assert!(
        msg.contains("lora"),
        "error should name the sink, got: {msg}"
    );
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
