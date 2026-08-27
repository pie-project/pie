//! The `masked` axis, end to end, and the three places it stops.
//!
//! **WHAT THIS FILE IS FOR.** `masked` is design §0's second supergraph axis
//! and the first one beyond decode/prefill: a per-lane fact the model
//! declares, a run-length mask on the submission, an `attention.masked` arm
//! over its own window. The bits path is written and unit-tested
//! (`driver_cuda::mask`); what nothing else in this tree can say is whether
//! the CATALOG can run it, and the answer today is no, for three reasons
//! that are each somebody's to fix and none of them the mask path's.
//!
//! So this file pins the reasons, against the real catalog, so that the day
//! one of them is fixed the test that was asserting the refusal fails and
//! says so. Every assertion here is host-side except the last, which needs a
//! device and skips without one.
//!
//! ```text
//! cargo test -p driver-cuda --test masked_axis
//! cargo test -p driver-cuda --features cuda-13 --test masked_axis -- --nocapture
//! ```

use driver::driver_api::fire::Mask;
use driver_cuda::{Fault, LaneMask};
use model_compiler::{Budgets, DeviceProfile, compile};
use model_dsl::Plane;
use model_ir::{Attention, Operation, Plan};

/// A deployment's ceilings, small: nothing here loads a checkpoint except the
/// last test, which states its own.
fn budgets() -> Budgets {
    Budgets::new(8, 512)
}

/// How many `attention.masked` arms a SKU's trace carries.
fn masked_arms(plan: &Plan) -> usize {
    plan.nodes
        .iter()
        .filter(|node| {
            matches!(
                node.op,
                Operation::Attention(Attention::Masked { .. })
            )
        })
        .count()
}

/// **GEMMA IS THE ONLY FAMILY THAT DECLARES THE AXIS.**
///
/// Stated rather than assumed, because every other claim in this file is
/// about gemma and a reader is entitled to know why: `masked` is a
/// model-declared fact (design §8), the bits are a runtime input, and a plan
/// with no `attention.masked` node has nowhere for them to go. A qwen lane
/// carrying a mask is not "the axis without the gemma facts" — it is a mask
/// nothing reads.
#[test]
fn gemma_is_the_only_family_that_declares_the_masked_axis() {
    let mut declaring: Vec<(String, usize)> = Vec::new();
    for (sku, _, trace, _) in model::catalog() {
        let arms = masked_arms(&trace(Plane::Cuda));
        if arms > 0 {
            declaring.push((sku.to_string(), arms));
        }
    }
    assert!(
        declaring.iter().all(|(sku, _)| sku.starts_with("gemma4-")),
        "a family beyond gemma declares `attention.masked`, and the gates in \
         this file were written against gemma alone: {declaring:?}"
    );
    assert!(
        !declaring.is_empty(),
        "no SKU declares `attention.masked` at all, and then the axis has no \
         model text to be exercised by"
    );
}

/// **BLOCKER 1: gemma's kv space carries two geometries, and `SpaceFacts`
/// holds one.**
///
/// Gemma alternates a 256-wide sliding attention with a 512-wide global one
/// and puts every layer's cache row in ONE space, so `kv::probe` — which
/// reads a space's head width, head count and window off the ops that name it
/// — finds two answers and refuses. Nothing about masks: this is why gemma
/// does not load AT ALL on this shell, and it stands in front of every gate
/// the masked axis would otherwise reach.
///
/// Pinned as a refusal so that the day the shell seats per-kind schedules,
/// this test fails and the axis's real gates can be written.
#[test]
fn gemma_does_not_load_because_one_kv_space_holds_two_geometries() {
    let plan = model::trace_of("gemma4-e4b-bf16-kv-bf16").expect("the catalog ships gemma")(
        Plane::Cuda,
    );
    let probed = driver_cuda::store::kv::probe(&plan);
    let Err(Fault::Unbound { what }) = probed else {
        panic!(
            "gemma's kv space probes cleanly now — the masked axis's first \
             blocker is gone and this file's gates can be written for real: \
             {probed:?}"
        );
    };
    assert!(
        what.contains("disagree about its shape"),
        "gemma refuses for a different reason than its two head widths: {what}"
    );
}

/// **BLOCKER 2: gemma's masked arm and its prefill arm share one schedule.**
///
/// `plan_p` is minted once in `gemma_4::forward` and read by both
/// `attention.prefill` and `attention.masked`, which stand in different
/// classes. The compiler narrows a prepare node by demand to the UNION of the
/// classes reading its struct (design build log 7) — right for a shared
/// value, wrong for two windowed readers — so the schedule is carved over
/// both classes and each arm hands it its own, narrower, rebased boundaries.
/// Every work item past the first request then indexes a `qo_indptr` that has
/// already ended.
///
/// The shell refuses it by name at LOAD rather than reading past a vector at
/// the fire. The fix is one line of model text: a second
/// `inputs.plan_prefill()` for the masked arm, whose only reader is in one
/// class and whose region therefore carries one.
#[test]
fn gemma_s_masked_arm_shares_a_schedule_with_its_prefill_arm() {
    let plan = model::trace_of("gemma4-e4b-bf16-kv-bf16").expect("the catalog ships gemma")(
        Plane::Cuda,
    );
    let baked = compile(&plan, &budgets(), &DeviceProfile::default()).expect("gemma bakes");
    let checked = driver_cuda::window::no_schedule_straddles_its_readers(&plan, &baked);
    let Err(Fault::Straddled {
        planned, consumed, ..
    }) = &checked
    else {
        panic!(
            "gemma's arms no longer share a schedule — the masked axis's second \
             blocker is gone: {checked:?}"
        );
    };
    assert_ne!(
        planned, consumed,
        "a straddle is two DIFFERENT class sets, and this one names the same twice"
    );
}

/// And nothing that serves today is newly refused by that check.
///
/// **THE OTHER HALF OF A NEW REFUSAL.** A load-time check that refuses a
/// defect nobody had is worth nothing if it also refuses the SKUs the shell
/// runs, so every catalog row is asked, and only gemma's may answer.
#[test]
fn no_other_sku_straddles_a_schedule() {
    let mut straddled: Vec<String> = Vec::new();
    for (sku, _, trace, _) in model::catalog() {
        let plan = trace(Plane::Cuda);
        let Ok(baked) = compile(&plan, &budgets(), &DeviceProfile::default()) else {
            continue;
        };
        if let Err(fault) = driver_cuda::window::no_schedule_straddles_its_readers(&plan, &baked) {
            straddled.push(format!("`{sku}`: {fault}"));
        }
    }
    assert!(
        straddled.iter().all(|line| line.starts_with("`gemma4-")),
        "a SKU beyond gemma straddles a schedule, and this check would refuse a \
         load that works today:\n{}",
        straddled.join("\n")
    );
}

/// **BLOCKER 3: there is no custom-mask kernel arm with a sliding window.**
///
/// `fa2::prefill_custom_arm` instantiates `Custom` and `CustomSoftcap` and no
/// windowed variant, and `attn::masked` refuses a schedule carved for a
/// window because the window would then ride the launch over a plan that does
/// not cover the prefix. Gemma states `Some(512)` on five layers out of every
/// six, so even with blockers 1 and 2 gone its masked arm has no kernel on
/// most of its stack.
///
/// It cannot be folded into the bits, either, and that is worth writing down:
/// a window is stated per NODE and the mask slab is per FIRE, so folding it
/// would mean one slab per layer window. Named here; the device text is not
/// this wave's to edit.
#[test]
fn gemma_states_a_sliding_window_on_the_arm_that_has_no_windowed_variant() {
    let plan = model::trace_of("gemma4-e4b-bf16-kv-bf16").expect("the catalog ships gemma")(
        Plane::Cuda,
    );
    let windowed = plan
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Attention(Attention::Masked { window, .. }) => Some(*window),
            _ => None,
        })
        .filter(Option::is_some)
        .count();
    assert!(
        windowed > 0,
        "gemma's masked arms state no window any more, and `fa2::prefill_custom_arm` \
         having no windowed variant has stopped mattering"
    );
    assert!(
        windowed < masked_arms(&plan),
        "every one of gemma's masked arms states a window, which would make the \
         axis entirely unreachable rather than partly"
    );
}

/// The bits a lane's runs expand to are the bits the device text addresses.
///
/// **ONE TEST THAT IS ABOUT THE ARITHMETIC AND NOT ABOUT A REFUSAL.** The
/// custom-mask variant reads `qo_idx * kv_len + kv_idx`, LSB-first inside
/// each byte, at `maybe_custom_mask + maybe_mask_indptr[batch_idx]` — a BYTE
/// offset on this device text, not the bit offset upstream flashinfer
/// carries. This walks the staged bytes exactly that way, over a fire whose
/// lanes are DIFFERENT lengths and whose masked lane is not the first, so the
/// span table is doing real work.
#[test]
fn the_staged_bits_read_back_the_way_the_device_text_addresses_them() {
    // Lane 0: unmasked, 3 held, 2 new. Lane 1: masked, 5 held, 3 new, with
    // positions 0 and 6 dropped. Lane 2: unmasked decode.
    let mask = Mask::new(vec![1, 5, 1, 1], 8);
    let staged = driver_cuda::mask::stage(&[
        LaneMask {
            mask: None,
            have: 3,
            rows: 2,
        },
        LaneMask {
            mask: Some(&mask),
            have: 5,
            rows: 3,
        },
        LaneMask {
            mask: None,
            have: 9,
            rows: 1,
        },
    ])
    .expect("the mask covers its lane")
    .expect("a masked fire stages bits");

    // Lane 1 is 3 x 8 = 24 cells = 3 bytes, and it is the only lane holding
    // any, so the table is 0,0,3,3.
    assert_eq!(staged.indptr, vec![0, 0, 3, 3]);
    assert_eq!(staged.bits.len(), 3);

    let base = staged.indptr[1] as usize;
    let kv = 8usize;
    for q in 0..3usize {
        for k in 0..kv {
            let cell = q * kv + k;
            let read = (staged.bits[base + cell / 8] >> (cell % 8)) & 1 == 1;
            // The runs keep 1..=5 and 7; the causal bound keeps k <= 5 + q.
            let want = ((1..=5).contains(&k) || k == 7) && k <= 5 + q;
            assert_eq!(read, want, "cell ({q}, {k}) of lane 1");
        }
    }
    // The one position both terms drop, from opposite directions: key 7 is
    // KEPT by the runs and reachable only by the last query row.
    let seven = |q: usize| {
        let cell = q * kv + 7;
        (staged.bits[base + cell / 8] >> (cell % 8)) & 1 == 1
    };
    assert!(!seven(0) && !seven(1) && seven(2));
}

/// A mask against a plan with no masked arm is refused BY NAME, at the fire.
///
/// **THE REFUSAL THIS WAVE REPLACED, AND WHY IT MOVED.** The shell used to
/// answer `Unsupported { verb: "explicit attention masks" }` for every model,
/// which said the CUDA plane could not carry a mask. It can: the bits stage,
/// the seats bind, the span table slices. What decides is the artifact, so
/// the refusal is now `Fault::Maskless` and it is asked against the loaded
/// plan.
///
/// Skips without a device and a checkpoint, like every other test in this
/// tree that needs one.
#[test]
fn a_mask_against_a_maskless_artifact_is_refused_by_name() {
    if !driver_cuda::device::present() {
        eprintln!("skipping the maskless refusal: no CUDA device on this machine");
        return;
    }
    let Some((mut shell, _)) = common::ready("the maskless refusal") else {
        return;
    };
    shell.open(0).expect("slot 0 opens");
    let tokens = [9707u32, 11, 1879];
    let mask = Mask::new(vec![0, 3], 3);
    let refused = shell.fire_seated(&[driver_cuda::Seated::masked(
        driver_cuda::Lane {
            slot: 0,
            word: common::word(tokens.len() as u32),
            tokens: &tokens,
        },
        &mask,
    )]);
    assert!(
        matches!(refused, Err(Fault::Maskless { lane: 0 })),
        "a masked lane against qwen — which bakes no `attention.masked` arm — \
         must be refused by name, not run unmasked: {refused:?}"
    );

    // And the same lane WITHOUT a mask still fires, so the refusal is about
    // the mask and not about the submission around it.
    let fired = shell.fire(&[driver_cuda::Lane {
        slot: 0,
        word: common::word(tokens.len() as u32),
        tokens: &tokens,
    }]);
    assert!(
        fired.is_ok(),
        "the same lane without a mask must still fire: {fired:?}"
    );
}

/// The load, shared with `serve_smoke` in shape and stated here rather than
/// imported because a test binary is its own crate.
mod common {
    use std::path::{Path, PathBuf};

    use driver_cuda::{Boot, Shell};
    use model_compiler::Budgets;
    use model_dsl::{Classify, Plane, Request};

    const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

    pub fn word(query_len: u32) -> u64 {
        model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
    }

    fn snapshot() -> Option<PathBuf> {
        if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
            let path = PathBuf::from(stated);
            return path.is_dir().then_some(path);
        }
        let home = std::env::var("HOME").ok()?;
        let snapshots =
            Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
        std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .find(|path| path.join("tokenizer.json").exists())
    }

    fn container(snapshot: &Path) -> Option<PathBuf> {
        let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
            .ok()?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?;
                (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
            })
            .collect();
        found.sort();
        found.into_iter().next()
    }

    pub fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
        let Some(checkpoint) = snapshot() else {
            eprintln!("skipping {what}: no Qwen3.5-0.8B snapshot (set PIE_SMOKE_SNAPSHOT)");
            return None;
        };
        let Some(container) = container(&checkpoint) else {
            eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
            return None;
        };
        let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
            .expect("the checkpoint's tokenizer loads");
        let plan = model::trace_of(SKU).expect("the catalog ships the SKU")(Plane::Cuda);
        let source = ztensor_compat::index(&container).expect("the checkpoint opens");
        let contract = model::import_of(SKU).expect("the catalog ships an import")(&source)
            .expect("the import contract fits its own checkpoint");
        drop(source);
        let shell = Shell::load(Boot {
            plan,
            contract: &contract,
            checkpoint: &checkpoint,
            budgets: Budgets::new(4, 256),
            profile: None,
            page_size: 16,
            context: 512,
            slots: 4,
            ordinal: 0,
            graphs: driver_cuda::Graphs::Off,
        })
        .expect("the shell loads");
        Some((shell, tokenizer))
    }
}
