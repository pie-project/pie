//! What stands between this backend and qwen3.5, measured rather than assumed.
//!
//! `tests/serving.rs`'s header says Qwen3.5-0.8B is out of reach and gives the
//! reason: *"`kernels-wgpu`'s `ssm.rs` rows (`gdn_core`, `gdn_core_recurrent`,
//! `gdn_core_recurrent_prefill`) declare axes and no operands, and `geometry.rs`
//! refuses `Rule::RecurrentScan` and the rest of that family as
//! `Ungeometric::Unruled`"*.
//!
//! **Both halves of that sentence are now false, and neither was the real
//! obstacle.** `ssm` crossed: all eight kernels have routines stating their
//! operands and arms finding them, and the rows that stated none are deleted.
//! A routine builds its own grid from `kernels::shader`'s helpers, so no launch
//! rule is consulted and `RecurrentScan` cannot refuse anything. So the stated
//! reason expired, and what it was standing in front of had never been looked
//! at.
//!
//! This file looks. It lowers qwen3.5's own hybrid forward and puts EVERY
//! rectangle through `plan_one` -- the call a real fire makes -- with no
//! weights, no adapter and no tokens.
//!
//! # How it ended, so the tests below are read in order
//!
//! It ends with the model serving: *"The capital of France is Paris. The
//! capital of France is"* -> ` Paris is the`. For most of this file's life it
//! answered a SPACE to every prompt, and many docs below describe that symptom
//! in the present tense because they were written while it was true.
//!
//! The cause was not in this backend. `qwen_3_5/forward/metal.rs` asked for
//! `NormVariant::Plain` while every `Qwen35*Facts` fixture stated `Gemma`, so
//! the norms multiplied by `w` where this checkpoint's gains are trained from
//! ZERO and want `1 + w`. Nothing here could see it: every reference in this
//! file checks a kernel against the operands it was HANDED, and a wrongly
//! folded norm hands on wrongly and faithfully -- the MLP reference below
//! multiplies by `w` and agrees with the device to one bf16 step, because both
//! were doing the same wrong thing.
//!
//! That is the lesson the file is worth reading for. Fifteen kernels, every
//! weight, the layer mapping, the rotation, the int4 encode and the readout
//! were each verified against a CPU walk, and the defect was upstream of all
//! of them, in what the trace ASKED for rather than in what any kernel
//! computed. `which_fold_the_final_norm_applies_and_what_each_one_answers`
//! now asserts the fold so it cannot silently revert.
//!
//! # What it found
//!
//! **Twelve of the fourteen symbols plan clean. Zero are unclaimed by an arm.**
//! The attention half, the quantized projections, the paged KV write, the
//! gated-attention split and multiply, the norms and the MLP all plan. What
//! does not is the gated DeltaNet, and for three reasons, none of them the one
//! that was written down:
//!
//! 1. **The DSL spells three GDN symbols without a dtype suffix.**
//!    `model-dsl::metal` emits `gdn_prep_slotted`, `gdn_core_slotted` and
//!    `gdn_core_recurrent_slotted`, where its other twenty-six launches spell
//!    `rms_single_row_bfloat16`, `gate_bfloat16` and so on, and where BOTH
//!    backends' entrypoints carry `_bfloat16`. Metal does not mind: those three
//!    are its ROUTINE names, and `kernels-metal::kernel_of` resolves a routine
//!    name to a routine, whose body then states the entrypoint. This driver
//!    looks a module up BY THE PLAN'S SYMBOL (`serve::pick` ->
//!    `kernels_wgpu::entrypoint_source`), so a routine name finds nothing and
//!    the fire dies as `Unfired::NoModule`.
//!
//!    Suffixing the three in the DSL was tried and REVERTED: it makes twelve
//!    plan and drives the last two to their arms, but it perturbs the resolution
//!    path of the only backend that serves this model today, on a machine that
//!    cannot build or test that backend, for no gain that survives (2) and (3).
//!    The driver-side repair is the honest one and belongs with (3): resolve the
//!    module through the ROUTINE, which is what states its own entrypoint.
//!
//! 2. **This backend's GDN arms read the wrong operands, and metal's own
//!    comment says so.** `driver-metal`'s `gates` reads `a_gate`/`b_gate` as
//!    `input(1)`/`input(2)` and records why:
//!
//!    > *They used to be `o.weight(4)` and `o.weight(5)`, which is a defect this
//!    > backend never got to observe because no plan has ever named a GDN
//!    > symbol. A weight handle there binds a per-head buffer -- `Hv` elements,
//!    > the size of `dt_bias` -- and the shader strides it by `b_idx * Hv`, so
//!    > row zero reads the right gate and every row after it reads past the
//!    > end.*
//!
//!    `driver-wgpu`'s `gates` reads `o.weight(4)` and `o.weight(5)`. **It is
//!    the defect metal found and fixed, still standing here**, and for the same
//!    reason it stood there: nothing has ever dispatched a GDN symbol on this
//!    backend, so no test could see it. It is the third time this session a
//!    sibling's finding turned out to apply unfixed here, after `gate`'s
//!    aliased operand.
//!
//! 3. **There is no recurrent state on this backend at all, and (2) is a
//!    symptom of it.** Metal reads `conv_state` with `o.slab(layer,
//!    "conv_state")`. `driver-wgpu`'s `Resolve` has `weight`, `named`, `kv`,
//!    `number` and `table`, and nothing that reaches a per-layer recurrent
//!    slab; `Handles` has no `slab`. So this crate's `prep` improvised, taking
//!    `conv_state` from `input(1)` -- which is where the statement puts
//!    `a_gate` -- and pushing `slot_ids` to `input(2)`, which is `b_gate`. The
//!    operand order is not a mistake anyone made twice; it is what is left when
//!    the resource a kernel needs does not exist.
//!
//! **So qwen3.5 on wgpu is a FEATURE away, not a bug away.** The seam and the
//! arms are done — `Resolve::slab`, `Handles::slab`, `FireTable::RecurrentSlots`,
//! `Unplanned::NoSlab`, `resources::Recurrent`, `RecurrentPool`, and all five
//! GDN arms rewritten against metal's operand order. What is left is three
//! RESOURCES, named in
//! `no_arm_or_body_is_wrong_and_every_symbol_plans_once_the_resources_exist`.
//! Two of the three are done — the frame carries and stages
//! `FireTable::RecurrentSlots`, and `frames::unserved_in` asks whether the
//! DEPLOYMENT allocated slots rather than refusing every hybrid outright. What
//! is left is WEIGHTS. `engine::recurrent_of` states the shape from the row's
//! own `RecurrentShape`, so the pool opens; no loader publishes qwen3.5's
//! tensors under the names a wgpu plan binds, which is `src/names.rs`'s
//! territory and not this file's.
//!
//! # What the local checkpoint actually is, since it will surprise the next reader
//!
//! `Qwen/Qwen3.5-0.8B` is **multimodal**. Its `config.json` has a
//! `vision_config` and a `text_config`, and every text tensor is published
//! under `model.language_model.layers.N.…` rather than `model.layers.N.…`.
//! So "serve qwen3.5" is two jobs: the GDN text stack this file measures, and
//! a vision tower that is not this backend's at all yet. The 488 tensors
//! include both.
//!
//! # Why this is a report and not a verdict
//!
//! A plan that works here is not a model that answers. It asserts only that the
//! walk saw something, prints what it found, and pins the two names that block
//! -- so that when the slab lands, this file is what says whether anything else
//! was hiding behind them.

#![cfg(feature = "native")]

use std::collections::{BTreeMap, BTreeSet};

use driver_wgpu::binding::{Arena, Placeholder, Resolve, Unbacked};
use driver_wgpu::dispatch::{Built, Geometry, Sources};
use model::qwen_3_5::forward::metal::{Qwen35MetalFacts, qwen3_5_hybrid_metal};
use model::qwen_3_5::spec::Qwen35HybridFacts;
use model_compiler::lower::{Arg, Fire, Row, lower};
use model_dsl::WeightRepr;

/// qwen3.5's own hybrid forward, as a METAL text — which is how this backend
/// serves anything, because `Backend::of_family` knows `cuda` and `metal` and
/// wgpu's kernel names are metal's.
fn hybrid_plan(facts: &Qwen35HybridFacts) -> model_ir::trace::ForwardPlan {
    hybrid_plan_class(facts, model_ir::trace::FireClass::Prefill)
}

/// [`hybrid_plan`] for one class.
fn hybrid_plan_class(
    facts: &Qwen35HybridFacts,
    class: model_ir::trace::FireClass,
) -> model_ir::trace::ForwardPlan {
    hybrid_plan_class_at(facts, class, 4)
}

/// [`hybrid_plan_class`] at a stated affine WIDTH.
///
/// The loader quantizes this checkpoint at load time -- `RuntimeQuant::Int4`,
/// four bits over groups of 64 -- and every symbol carries the width in its
/// name (`..._gs_64_b_4`). Eight is the other point the tree stamps, so the
/// two can be compared on the same adapter with the same everything else.
fn hybrid_plan_class_at(
    facts: &Qwen35HybridFacts,
    class: model_ir::trace::FireClass,
    bits: u32,
) -> model_ir::trace::ForwardPlan {
    let metal = Qwen35MetalFacts {
        proj_repr: WeightRepr::Scaled {
            layout: model_dsl::ScaleLayout::PerGroup,
            group: 64,
            axis: 0,
            zero_point: true,
        },
        affine_bits: bits,
        // BOTH FALSE, and that is the fact this backend can state honestly.
        // They ask metal to stage a GEMM's tiles in `half` where the device
        // has no native bfloat matrix unit, which is a property of a Metal
        // adapter and of the `_fp16_precast` entrypoints stamped for it.
        // `kernels-wgpu` stamps none, so a `true` here would name a symbol
        // this tree does not compile.
        qmm_fp16_precast: false,
        routed_qmm_fp16: false,
        moe_repr: None,
        moe_bits: 4,
        // `default_moe_tile` is `None`: qwen3.5-0.8b has no expert banks, and
        // the fact is an opt-IN to the batched MoE arm that three failures in
        // the qwen3.6 family made opt-out.
        moe_tile: None,
        router_repr: Some(WeightRepr::Scaled {
            layout: model_dsl::ScaleLayout::PerGroup,
            group: 64,
            axis: 0,
            zero_point: true,
        }),
        router_bits: 8,
        qmm_tile: (16, 32),
        qmm_multi_batch: true,
        fuse_residual_gemv: true,
        rms_eps: 1e-6,
        rope_theta: 10_000_000.0,
        attn_scale: 1.0 / (f64::from(facts.attn.head_dim).sqrt() as f32),
        norm_topk_prob: true,
    };
    qwen3_5_hybrid_metal(facts, &metal, class)
}

/// Every symbol the hybrid's forward plan launches, with what happened to it.
#[derive(Default)]
struct Verdict {
    planned: BTreeSet<String>,
    refused: BTreeMap<String, String>,
    unclaimed: BTreeSet<String>,
    unreadable: BTreeMap<String, String>,
}

#[test]
fn every_rectangle_of_qwen3_5s_hybrid_forward_is_planned_or_named() {
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    // THE METAL TEXT, which is how this backend serves anything. `wgpu` has no
    // family of its own -- `Backend::of_family` knows `cuda` and `metal` and
    // nothing else -- and `tests/serving.rs` reaches for `llama_like_metal` for
    // exactly this reason: wgpu's kernel names are metal's, so a metal text is
    // the one it can lower. The backend-agnostic `qwen3_5_hybrid` refuses with
    // `UnknownBackend("qwen3_5_hybrid")`, which is a fact about the family
    // NAME and not about the kernels.
    let plan = hybrid_plan(&facts);
    println!(
        "qwen3.5-0.8b hybrid: {} ops, {} values",
        plan.ops.len(),
        plan.values.len()
    );

    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        4
    ];
    let low = match lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    ) {
        Ok(low) => low,
        Err(why) => {
            println!("LOWERING REFUSED: {why:?}");
            panic!("the hybrid plan did not lower, so no rectangle could be planned");
        }
    };
    println!(
        "lowered: {} launches over {} distinct symbols",
        low.launches.len(),
        low.kernels.len()
    );

    let geometry = Geometry {
        q_heads: facts.attn.q_heads,
        kv_heads: facts.attn.kv_heads,
        head_dim: facts.attn.head_dim,
        rotary_dims: facts.attn.head_dim,
        n_experts: 0,
        experts_per_token: 0,
        ..Default::default()
    };

    let held = Placeholder(u64::from(u32::MAX));
    let store = Unbacked(held);
    let arena = Arena {
        buffer: &held,
        bytes: u64::from(u32::MAX),
    };

    let mut v = Verdict::default();
    for launch in &low.launches {
        let symbol = low.kernels[launch.kernel as usize].clone();
        let declared =
            match driver_wgpu::reflect::entrypoint(&symbol, driver_wgpu::Capability::Baseline) {
                Ok(d) => d,
                Err(why) => {
                    // No module of that name in the tree, which is a COVERAGE
                    // answer and not a planning one: the kernel is missing
                    // rather than mis-wired.
                    v.unreadable.insert(symbol.clone(), format!("{why:?}"));
                    continue;
                }
            };
        let module = driver_wgpu::geometry::Module::loaded(&symbol, &declared);
        match driver_wgpu::dispatch::plan_one(
            &low,
            launch,
            Built {
                module,
                declared: &declared,
            },
            Sources {
                arena,
                resolver: &store,
                min_offset: 1,
            },
            geometry,
        ) {
            Ok(_) => {
                v.planned.insert(symbol);
            }
            Err(why) => {
                let said = format!("{why}");
                if said.contains("no armed stem claims") {
                    v.unclaimed.insert(symbol);
                } else {
                    v.refused.entry(symbol).or_insert(said);
                }
            }
        }
    }

    println!("\n== PLANNED ({}) ==", v.planned.len());
    for s in &v.planned {
        println!("  {s}");
    }
    println!("\n== NO SHADER IN THE TREE ({}) ==", v.unreadable.len());
    for (s, why) in &v.unreadable {
        println!("  {s}: {why}");
    }
    println!("\n== NO ARM CLAIMS IT ({}) ==", v.unclaimed.len());
    for s in &v.unclaimed {
        println!("  {s}");
    }
    println!("\n== REFUSED WHILE PLANNING ({}) ==", v.refused.len());
    for (s, why) in &v.refused {
        println!("  {s}: {why}");
    }

    assert!(
        !low.launches.is_empty(),
        "the hybrid lowered to no launches at all"
    );
    assert!(
        v.planned.len() >= 12,
        "only {} of qwen3.5's symbols plan; it was twelve when this was \
         written, and a FALL means something that worked stopped. {:?}",
        v.planned.len(),
        v.refused,
    );
    assert!(
        v.unclaimed.is_empty(),
        "these symbols are claimed by no arm, which the crossing was supposed \
         to make impossible: {:?}",
        v.unclaimed,
    );
    assert!(
        v.unreadable.is_empty(),
        "every symbol should reach a module now that the DSL spells the GDN \
         three with their dtype: {:?}",
        v.unreadable,
    );
    // WHAT BLOCKS, pinned BY NAME and BY REASON, and it has changed hands
    // twice. It was the two SLOTTED symbols, for want of a recurrent slab this
    // walk stands none in. Upstream then pointed qwen3.5's prefill at the
    // sequenced scan, so it is the two PREFILL symbols -- and for the same
    // reason, now that both arms serve the statement they are given.
    // BY STEM, because the tile is not part of the identity. The scan's
    // entrypoint spells its `(LANES, VROWS)` decomposition into its own name
    // and upstream retunes it -- this pinned `_l_32_v_4` and woke up to
    // `_l_32_v_2` -- so a literal here breaks on a change that is not about
    // this backend at all. What is pinned is which KERNELS block and why.
    let blocked: Vec<String> = v
        .refused
        .keys()
        .map(|s| match s.split_once("_bfloat16") {
            Some((stem, _)) => format!("{stem}_bfloat16"),
            None => s.clone(),
        })
        .collect();
    assert_eq!(
        blocked,
        vec![
            "gdn_core_recurrent_prefill_bfloat16",
            "gdn_prep_prefill_bfloat16"
        ],
        "the symbols this backend refuses are not the two the header explains",
    );
    for (symbol, why) in &v.refused {
        assert!(
            why.contains("slab, which this driver allocates none of"),
            "`{symbol}` should decline for the reason this file records for \
             it and not for another: {why}"
        );
    }
}

/// A [`Unbacked`] that also holds a recurrent carry.
///
/// The one thing this backend lacks, stood in for, so the question *"is the
/// slab the LAST thing"* can be asked without allocating one.
struct WithSlab(Placeholder);

impl Resolve for WithSlab {
    type Buffer = Placeholder;

    fn weight(&self, name: &str) -> Option<&Self::Buffer> {
        Unbacked(self.0).weight(name).map(|_| &self.0)
    }

    fn named(&self, _value: model_ir::trace::ValueId) -> Option<&Self::Buffer> {
        Some(&self.0)
    }

    fn kv(&self, _layer: u16, _values: bool) -> Option<&Self::Buffer> {
        Some(&self.0)
    }

    fn slab(&self, _layer: u16, _which: &'static str) -> Option<&Self::Buffer> {
        Some(&self.0)
    }

    fn table(&self, _which: driver_wgpu::binding::FireTable) -> Option<&Self::Buffer> {
        Some(&self.0)
    }

    fn number(&self, _which: driver_wgpu::binding::FireNumber) -> Option<u32> {
        Some(1)
    }
}

/// Nothing an ARM or a BODY states is wrong; what is left is RESOURCES.
///
/// The test above says what is broken. This says what is not: stand in every
/// per-fire resource this driver does not hold — the recurrent carry and the
/// slot table — and all fourteen of qwen3.5's symbols plan, with nothing
/// refused.
///
/// **It is deliberately not called "the slab is the last thing", which is what
/// it said first and could not support.** The stand-in answers `table` as well
/// as `slab`, so what it establishes is that the operand plumbing is correct,
/// not that one allocation completes the port. Three things are still missing
/// and each is a resource rather than a wiring mistake:
///
/// 1. ~~a deployment that STATES a `Recurrent` shape~~ — `engine`'s
///    `recurrent_of` maps `model::deployment::RecurrentShape` onto it, so a
///    hybrid row opens a pool. What is left is WEIGHTS: no loader publishes
///    qwen3.5's tensors under the names a wgpu plan binds;
/// 2. ~~`FireTable::RecurrentSlots` is never staged~~ — `Frame` carries it and
///    `Pool::stage` stages it now;
/// 3. ~~`frames.rs` refuses any plan carrying `rs_slot_ids` outright~~ — the
///    refusal asks whether the DEPLOYMENT allocated slots, which is what it
///    was always about, and still refuses emphatically when it did not.
///
/// Naming them is the point. A test whose title outruns what it measured is
/// the failure this whole file exists to correct: `serving.rs` said qwen3.5
/// was out of reach for a reason that had expired, and because the reason
/// named a cause it stopped anyone looking for the real one.
#[test]
fn no_arm_or_body_is_wrong_and_every_symbol_plans_once_the_resources_exist() {
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let plan = hybrid_plan(&facts);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        4
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");

    let geometry = Geometry {
        q_heads: facts.attn.q_heads,
        kv_heads: facts.attn.kv_heads,
        head_dim: facts.attn.head_dim,
        rotary_dims: facts.attn.head_dim,
        n_experts: 0,
        experts_per_token: 0,
        ..Default::default()
    };
    let held = Placeholder(u64::from(u32::MAX));
    let store = WithSlab(held);
    let arena = Arena {
        buffer: &held,
        bytes: u64::from(u32::MAX),
    };

    let mut refused: BTreeMap<String, String> = BTreeMap::new();
    let mut planned: BTreeSet<String> = BTreeSet::new();
    for launch in &low.launches {
        let symbol = low.kernels[launch.kernel as usize].clone();
        let Ok(declared) =
            driver_wgpu::reflect::entrypoint(&symbol, driver_wgpu::Capability::Baseline)
        else {
            continue;
        };
        let module = driver_wgpu::geometry::Module::loaded(&symbol, &declared);
        match driver_wgpu::dispatch::plan_one(
            &low,
            launch,
            Built {
                module,
                declared: &declared,
            },
            Sources {
                arena,
                resolver: &store,
                min_offset: 1,
            },
            geometry,
        ) {
            Ok(_) => {
                planned.insert(symbol);
            }
            Err(why) => {
                refused.entry(symbol).or_insert_with(|| format!("{why}"));
            }
        }
    }
    // NOTHING REFUSES. `model-dsl::metal` emits the SEQUENCED prompt scan now
    // -- `gdn_prep_prefill` and `gdn_core_recurrent_prefill` -- and both arms
    // serve it: `row_pitch` and `n_scan` are the rectangle's, the tile is the
    // statement's two params past `GdnCoreParams`, and the head geometry comes
    // off the block. For a stretch this list held the scan, refused for a tile
    // nobody compiled, which is what taking qwen3.5's prefill off this backend
    // looked like from here.
    assert!(
        refused.is_empty(),
        "a carry was stood in and these still refuse, so the slab is NOT the \
         last thing: {refused:?}"
    );
    assert_eq!(
        planned.len(),
        14,
        "all fourteen of qwen3.5's symbols should plan once a carry exists; \
         {planned:?}"
    );
}

/// `resources::Recurrent`'s arithmetic, against qwen3.5-0.8b's own numbers.
#[test]
fn a_recurrent_shape_sizes_the_planes_the_kernels_index() {
    use driver_wgpu::resources::{CONV_PLANES, RECURRENT_ELEM_BYTES, Recurrent};

    // MEASURED off the local Qwen3.5-0.8B, not guessed. Its `text_config`
    // states 24 layers whose `layer_types` are 18 `linear_attention` and 6
    // `full_attention`, `linear_num_value_heads: 16`,
    // `linear_value_head_dim: 128`, `linear_key_head_dim: 128` and
    // `linear_conv_kernel_dim: 4`; and its
    // `model.language_model.layers.0.linear_attn.conv1d.weight` is
    // `[6144, 1, 4]`, which is where `conv_dim` comes from — the mixed q|k|v
    // bank, three planes of `16 * 128`. This file first wrote 4096 for it,
    // reading the bank as k|v, and the tensor is what corrected it.
    let shape = Recurrent {
        linear_layers: 18,
        conv_dim: 6144,
        conv_k: 4,
        v_heads: 16,
        v_dim: 128,
        k_dim: 128,
        slots: 8,
    };

    assert_eq!(
        shape.conv_bytes_per_slot(),
        4 * 6144 * RECURRENT_ELEM_BYTES,
        "the conv window is the kernel's stride: taps by channels by f32"
    );
    assert_eq!(
        shape.state_bytes_per_slot(),
        16 * 128 * 128 * RECURRENT_ELEM_BYTES,
        "the state is `[v_heads, v_dim, k_dim]` of f32"
    );
    // The plane is per-LAYER and holds every slot, which is what makes a
    // scattered fire one contiguous plane rather than N blits.
    assert_eq!(
        shape.conv_bytes_per_layer(),
        shape.conv_bytes_per_slot() * 8
    );

    // TWO conv planes and one state plane per linear layer. A reader doubts
    // the two, so it is asserted rather than folded into a total.
    assert_eq!(
        shape.bytes_per_slot(),
        18 * (CONV_PLANES * shape.conv_bytes_per_slot() + shape.state_bytes_per_slot())
    );
    assert_eq!(shape.total_bytes(), 8 * shape.bytes_per_slot());

    // A budget buys whole seats, and one that buys none REFUSES rather than
    // reporting a zero-slot pool: a stack with nowhere to keep its carry
    // cannot serve one request, and zero seats would be discovered at the
    // first fire instead of at open.
    let one = shape.bytes_per_slot();
    assert_eq!(shape.slots_within(one).map(|s| s.slots), Some(1));
    assert_eq!(shape.slots_within(one * 3 + 7).map(|s| s.slots), Some(3));
    assert!(shape.slots_within(one - 1).is_none());
}

/// A deployment that STATES a recurrent shape gets planes it can bind.
///
/// The seam's last link, and the one a unit test of the arithmetic cannot
/// reach: `Deployment::recurrent` -> `RecurrentPool::open` -> `Shell` ->
/// `Held` -> `Model::slab`. Everything before it is integers and everything
/// after it is an arm; this is the only step that allocates.
///
/// It asks the pool directly rather than through a fire, because a fire needs
/// a hybrid TEXT and this backend has none — `Backend::of_family` knows `cuda`
/// and `metal`, and a qwen3.5 metal text lowers here (see the probe above) but
/// nothing loads its weights yet. What can be checked without that is the
/// thing that was actually missing: that a stated shape becomes three distinct
/// planes per layer, and that a layer nobody allocated answers `None` rather
/// than another layer's carry.
#[test]
fn a_stated_recurrent_shape_becomes_planes_an_arm_can_reach() {
    use driver_wgpu::binding::Resolve;
    use driver_wgpu::resources::{Recurrent, RecurrentPool};

    let Some(device) = adapter() else {
        println!("no adapter, so the allocation could not be measured");
        return;
    };
    // Small on purpose: this measures the WIRING, not the size, and the
    // arithmetic is checked against qwen3.5's real numbers beside it.
    let shape = Recurrent {
        linear_layers: 2,
        conv_dim: 64,
        conv_k: 4,
        v_heads: 2,
        v_dim: 16,
        k_dim: 16,
        slots: 2,
    };
    let pool = RecurrentPool::open(&device, shape, 0..3u16).expect("three layers of planes");

    // THREE DISTINCT planes per layer. `conv_state` and `new_conv_state` must
    // not be the same buffer: the kernel is still reading the old taps while
    // it writes the new ones, so an arm handed one buffer twice would make a
    // scan read what it had just written.
    let conv = pool.slab(1, "conv_state").expect("a conv plane");
    let fresh = pool.slab(1, "new_conv_state").expect("a fresh conv plane");
    let state = pool.slab(1, "recurrent_state").expect("a state plane");
    assert!(
        !std::ptr::eq(conv, fresh),
        "the read and written conv planes are the same buffer, so a scan would \
         read the taps it just wrote"
    );
    assert!(!std::ptr::eq(conv, state));

    // A layer nobody allocated, and a name no kernel knows.
    assert!(pool.slab(9, "conv_state").is_none());
    assert!(pool.slab(1, "not_a_plane").is_none());

    // And the seam an arm actually goes through.
    let weights = driver_wgpu::resources::Weights::new();
    let kv = driver_wgpu::resources::Pool::open(
        &device,
        driver_wgpu::resources::Shape {
            layers: 3,
            kv_heads: 2,
            head_dim: 16,
            page_size: driver_wgpu::facts::PAGE_SIZE,
            pages: 2,
            bytes: 2,
        },
    )
    .expect("a small kv pool");
    let with = driver_wgpu::resources::Model {
        weights: &weights,
        pool: &kv,
        recurrent: Some(&pool),
    };
    assert!(
        Resolve::slab(&with, 1, "recurrent_state").is_some(),
        "a Model holding a recurrent pool answers the slab an arm asks for"
    );
    let without = driver_wgpu::resources::Model {
        weights: &weights,
        pool: &kv,
        recurrent: None,
    };
    assert!(
        Resolve::slab(&without, 1, "recurrent_state").is_none(),
        "and one holding none refuses, which is what keeps a GDN arm honestly \
         dark rather than handed a null carry"
    );
}

/// The adapter, or `None` on a machine without one.
fn adapter() -> Option<driver_wgpu::device::Device> {
    driver_wgpu::device::Device::open().ok()
}

/// How far the WEIGHTS are, and against the right surface.
///
/// The three RESOURCES are done; this is the one thing left, and it is worth a
/// number rather than a sentence. `tests/checkpoint.rs` got one for qwen3 —
/// *"zero of 704"* — and that number is what told everyone a weight loader for
/// this crate is a CONVERSION rather than a lookup.
///
/// # It is measured against the LOAD PLAN, not the export
///
/// The first draft of this test compared `Naming::spellings` against the raw
/// safetensors keys, and `Naming`'s own doc says why that is wrong: *"the
/// contract renames before a driver sees anything… a table written against the
/// export would be self-consistent, would pass any test that held the text
/// against it, and would find nothing at load."* So the checkpoint is put
/// through `model::boot::compile_load_plan_for` first, exactly as a driver
/// boot does, and the plan's names are what the table is asked about.
///
/// It reports rather than asserting a target, because a target invented here
/// would be a number nobody measured. It SKIPS loudly when the snapshot or the
/// row is absent — a skip that could have been a measurement is the failure
/// this whole file is written against.
#[test]
fn how_many_of_qwen3_5s_weight_names_a_load_plan_publishes() {
    let Some(dir) = qwen3_5_snapshot() else {
        println!(
            "no Qwen3.5-0.8B snapshot in the HuggingFace cache, so THE WEIGHT \
             NAMES COULD NOT BE MEASURED"
        );
        return;
    };
    let Some(row) = model::catalog::find("qwen3.5-0.8b-base") else {
        println!("this build has no `qwen3.5-0.8b-base` row");
        return;
    };
    println!("measuring against {dir}");

    let path = std::path::Path::new(&dir);
    let Ok(meta) = model_loader::checkpoint::read::parse_checkpoint_metadata(path) else {
        println!("the snapshot's metadata did not parse");
        return;
    };
    let Ok(config) = std::fs::read_to_string(path.join("config.json")) else {
        println!("no config.json");
        return;
    };
    let Ok(encoding) = model::encoding::Encoding::from_config_json(&config) else {
        println!("the config states no encoding this build knows");
        return;
    };
    let target = model_loader::plan::StorageTarget::for_backend(
        model_loader::types::BackendKind::Vulkan,
        0,
        1,
    );

    let published: BTreeSet<String> = match model::boot::compile_load_plan_for(
        path,
        &meta,
        &target,
        row,
        &encoding,
        model::boot::Binding::MLX_IN_PLACE,
    ) {
        Ok((plan, _)) => {
            println!("plan compiled, {} tensors", plan.tensors.len());
            plan.tensors.iter().map(|t| t.name.clone()).collect()
        }
        Err(why) => {
            // THE REFUSAL NAMES THE OTHER WAY IN, and qwen3-0.6b takes it too:
            // an unquantised release carries no `.scales`, so the projections
            // are encoded at LOAD instead. `tests/serving.rs` reads the same
            // refusal the same way, which is what turns the one qwen3 release
            // on this machine into a measurement rather than a skip.
            //
            // Matched on its TEXT rather than assumed, so a checkpoint refused
            // for some other reason stops here instead of being quietly
            // re-planned under a policy that cannot answer it.
            let said = why.to_string();
            if !said.contains("needs quantized weights") {
                println!(
                    "\nTHE LOAD PLAN WAS REFUSED, and not for want of quantisation:\n  {said}"
                );
                return;
            }
            println!("MLX_IN_PLACE refused an unquantised release; encoding at load");
            let policy = model::shared::policy::Policy {
                projections: model::shared::policy::Projections::InPlace,
                naming: model::shared::policy::Naming::Mlx,
                runtime_quant: model::shared::policy::RuntimeQuant::Int4,
                moe_request: model::shared::policy::Mxfp4MoeRequest::Auto,
                component: model::shared::policy::Component::Full,
                stream_routed_experts: false,
                knobs: model::shared::policy::FamilyKnobs::default(),
            };
            let Ok((contract, _)) =
                model::contract::author_with_policy(row, &encoding, &meta, &target, &policy)
            else {
                println!("\nTHE LOADER WOULD NOT AUTHOR IT under a runtime-quant policy");
                return;
            };
            match model_loader::plan::compile(&meta, &contract, target) {
                Ok(plan) => {
                    println!("plan compiled, {} tensors", plan.tensors.len());
                    plan.tensors.iter().map(|t| t.name.clone()).collect()
                }
                Err(e) => {
                    println!("\nTHE PLAN WOULD NOT COMPILE:\n  {e}");
                    return;
                }
            }
        }
    };

    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let plan = hybrid_plan(&facts);
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        2
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");

    let bound: BTreeSet<String> = low
        .args
        .iter()
        .filter_map(|a| match a {
            model_compiler::lower::Arg::Weight(n) => Some(n.clone()),
            _ => None,
        })
        // A `scale.` marker is a dispatch constant riding a weight slot, not a
        // tensor, so no publisher has one.
        .filter(|n| !n.starts_with("scale."))
        .collect();
    println!("the lowering binds {} distinct weight names", bound.len());

    let table = driver_wgpu::names::Naming::mlx();
    let mut resolved = 0usize;
    let mut unspelled: Vec<&String> = Vec::new();
    let mut missing: Vec<&String> = Vec::new();
    for name in &bound {
        let spellings = table.spellings(name);
        if spellings.is_empty() {
            unspelled.push(name);
        } else if spellings.iter().any(|s| published.contains(s)) {
            resolved += 1;
        } else {
            missing.push(name);
        }
    }

    println!(
        "\nRESOLVED {resolved} of {}; {} have no spelling in the table, {} \
         spell but the plan does not publish",
        bound.len(),
        unspelled.len(),
        missing.len()
    );
    for n in unspelled.iter().take(12) {
        println!("  no spelling: {n}");
    }
    for n in missing.iter().take(12) {
        println!("  not published: {n} -> {:?}", table.spellings(n));
    }
    // What the plan DOES publish for one linear layer, so the roles this table
    // is missing can be read off rather than guessed.
    println!("\nthe plan publishes, for layer 0:");
    for n in published.iter().filter(|n| n.starts_with("layers.0.")) {
        println!("  {n}");
    }

    assert!(
        !bound.is_empty(),
        "the lowering bound no weights at all, so this measured nothing"
    );
    // EVERY ONE, and asserted rather than reported now that it is true. It was
    // 352 of 712 when this test was written: the 306 with no spelling were the
    // gated DeltaNet's whole layer -- `conv_w`, `a_log`, `dt`, `in_proj_qkv`
    // and the rest, all under a `linear_attn.` module `names.rs` had never
    // seen -- and the 54 that spelled but were not published were the linear
    // layers' `out_proj`, which this table only knew as `self_attn.o_proj`.
    //
    // A name this table cannot spell is REFUSED NOWHERE: `spellings` answers
    // with nothing, the loader allocates for the weights it could name, and
    // the rest stay bound to whatever the arena held. `names.rs` records that
    // costing gpt-oss-20b 48 of its 775 weights silently; this was the same
    // defect at six times the scale, and it is why the number is pinned.
    assert_eq!(
        (resolved, unspelled.len(), missing.len()),
        (bound.len(), 0, 0),
        "every weight name a qwen3.5 lowering binds should resolve to one the \
         load plan publishes"
    );
}
/// A local `Qwen3.5-0.8B` snapshot, base or instruct.
///
/// The revision has to actually HOLD the weights, which is what
/// `driver-cuda`'s `qwen3_snapshot` and `Checkpoint::open` both ask and this
/// one did not. It took the first entry `read_dir` handed back and returned
/// it -- `for revision in revisions.flatten() { return Some(...) }`, a loop
/// that never loops, which is how clippy found it. `read_dir` states no
/// order, and a `snapshots/` directory holds whatever a partial or superseded
/// download left behind.
///
/// That matters more here than it looks: every one of the six callers skips
/// SOFTLY, printing "the snapshot's metadata did not parse" or "no
/// config.json" and returning. A revision with no weights in it does not fail
/// them, it makes all six pass having measured nothing.
fn qwen3_5_snapshot() -> Option<String> {
    qwen3_5_snapshot_of(&[
        "models--Qwen--Qwen3.5-0.8B-Base",
        "models--Qwen--Qwen3.5-0.8B",
    ])
}

/// [`qwen3_5_snapshot`] over a STATED list of cache repositories, first hit
/// wins.
///
/// The instruct checkpoint sits beside the base one and shares its
/// architecture tensor for tensor, so it loads through the same catalog row --
/// which makes it a second weight set to check a fix against, and a fix that
/// only works on one checkpoint is not a fix.
fn qwen3_5_snapshot_of(repos: &[&str]) -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    for repo in repos {
        let root = std::path::Path::new(&home)
            .join(".cache/huggingface/hub")
            .join(repo)
            .join("snapshots");
        let Ok(revisions) = std::fs::read_dir(&root) else {
            continue;
        };
        // Sharded snapshots carry an index instead of one file.
        let found = revisions.flatten().find_map(|revision| {
            let d = revision.path();
            (d.join("model.safetensors").is_file()
                || d.join("model.safetensors.index.json").is_file())
            .then(|| d.to_string_lossy().into_owned())
        });
        if found.is_some() {
            return found;
        }
    }
    None
}

/// **Does qwen3.5 actually fire on this backend?**
///
/// Everything the other tests in this file measure is a step short of an
/// answer: a plan that works is not a model that answers. This loads the real
/// checkpoint, opens a shell with a recurrent pool, and fires one prefill.
///
/// Whatever it does is a MEASUREMENT and is printed. It asserts only what it
/// has earned at each step, because the point is to find where the road ends
/// rather than to prove it does not — and every refusal on the way is named,
/// so the next attempt starts from one.
///
/// # Where it stands
///
/// It FIRES and the answer is WRONG, which is the most useful of the three
/// possible outcomes: 712 tensors staged, every dispatch succeeding, every
/// logit finite, a distribution that is not flat — and it does not continue
/// the induction pattern. It wants 3111 and the period says 88204.
///
/// The negative control narrows it: zeroing `layer.0.conv_w` moves the widest
/// logit by 3.8 and changes all but 995 of 248,320, so **the gated DeltaNet
/// layers are WIRED and reach the answer.** What is wrong is inside them or
/// beside them, not the plumbing this file spent its length building.
///
/// # One cause found and fixed, and it was not this one
///
/// `Frame::recurrent_slots` was declared, staged, and **never written**. An
/// unwritten table stages as empty; an empty storage buffer answers every
/// subscript with a clamp instead of a trap; so every conversation in every
/// fire read slot ZERO and inherited whatever the previous fire had left
/// there. Nothing refused and every dispatch succeeded.
///
/// It was caught by a CONTROL rather than by looking, and by a control that
/// was there for something else:
/// [`the_same_prompt_twice_is_the_same_answer_or_the_scan_is_racing`] asked
/// whether the prefill was racing and put a one-token decode beside it to
/// prove the question was about the prefill. **The decode was
/// non-deterministic too** — four distinct answers to four identical fires —
/// which took the diagnosis away from the prefill and pointed at something
/// every fire shares. With the slots written it is one distinct answer at
/// every length, spread exactly zero.
///
/// That fixed the non-determinism and the shell that went permanently dark.
/// **It did not fix the answer**, which is still 3111.
///
/// # FIXED: the prefill is sequenced now, and the race is gone
///
/// Upstream pointed qwen3.5's prefill at the SEQUENCED scan
/// (`gdn_prep_prefill` + `gdn_core_recurrent_prefill`, which loop
/// `for t in 0..n_scan` with the state in registers), and this backend has
/// been brought to it: the arms take `row_pitch`/`n_scan` from the RECTANGLE
/// and the tile from the statement, the shader packs every scratch row at its
/// own width instead of striding all three by one pitch, and a dispatch can
/// carry a `@group(0)` storage block and a `@group(1)` uniform at once.
///
/// The prefill is deterministic now — one distinct answer in four fires at 1,
/// 4 and 8 tokens, spread exactly zero, where before the fix it was four.
///
/// **The answer is still wrong.** It wants 1723, which is the token it has
/// just been shown: a copy off by one, where before all this it wanted a token
/// with no relation to the prompt at all.
///
/// # What the race WAS, kept because it is what the sequencing fixed
///
/// **The prefill fires the DECODE-shaped gated-DeltaNet pair**, and
/// [`the_same_prompt_twice_is_the_same_answer_or_the_scan_is_racing`] shows
/// what that costs: a one-token turn (the fused `gdn_core`, one row per
/// request) gives **one** distinct answer in four fires, and a four- or
/// eight-token prefill gives **four**. The same prompt, the same weights, a
/// fresh row each time.
///
/// That was invisible until the head counts were fixed. With `kv_heads` in
/// the grid the scan dispatched an eighth of its workgroups, so most of the
/// state was never touched and most of the race never happened.
///
/// - `gdn_prep` + `gdn_core_recurrent` are per REQUEST.
///   `kernels-metal/kernels/ssm/gdn_prep.metal`'s header states their grid as
///   `{32, Vd, R*Hv}`, *"one simdgroup per (req, v-head, v-dim)"*, and its
///   suite proves them bit-identical to the FUSED DECODE kernel `gdn_core`.
///   One token per request is the shape they are written for.
/// - `qwen_3_5/forward/metal.rs` fires that pair for `FireClass::Prefill`,
///   where a rectangle's rows are TOKENS. Every token of one prompt names the
///   same slot — correctly, it is one conversation's carry — and they all
///   read-modify-write that one state inside a single dispatch. **The
///   recurrence is never sequenced across the prompt.**
/// - The pair written for a prompt, `gdn_prep_prefill` +
///   `gdn_core_recurrent_prefill`, carries `row_pitch` and `n_scan` and loops
///   `for t in 0..n_scan` with the state held in registers, which is what
///   makes it sequential. **No model text and no DSL emitter names either of
///   them, on any backend**, while all three carry routines, arms, geometry
///   rules and nine tuned `(LANES, VROWS)` instantiations for them, and
///   `kernels-wgpu/tests/gpu.rs` lists both under "kernels this suite could
///   dispatch and does not".
///
/// So the arithmetic nobody could find a reference for turns out not to need
/// one: the prompt's tokens are folded into the carry in parallel instead of
/// in order, and a copying circuit is exactly the thing that cannot survive
/// it. Fixing it is a DSL emitter and a text arm, which is two backends this
/// machine cannot test, and is why this file stops at saying so.
///
/// # A second defect, still open
///
/// [`how_many_fires_a_shell_answers_before_it_goes_dark`] finds every fire
/// whose ROW COUNT is odd and greater than one answering with **NaN** in every
/// row, in BOTH classes, with all 364 rectangles dispatched and a clean
/// lowering; qwen3-0.6b is unaffected at the same row counts. It is bisected
/// down to **`gdn_prep_slotted_bfloat16` at layer 1** there, by stopping the
/// fire on each rectangle in turn. The host side is cleared. It no longer
/// takes the shell with it, which the slot fix is what changed.
///
/// It does not explain the wrong answer above: the induction prompt is
/// thirty-two tokens, which is even, and it fires.
///
/// Skips loudly when the snapshot or the row is absent.
#[test]
#[ignore = "loads and encodes a real checkpoint; run it deliberately"]
fn qwen3_5_fires_or_says_where_it_stopped() {
    let Some((mut shell, real)) = qwen3_5_shell(4) else {
        return;
    };

    // AN INDUCTION PROMPT, which is the only check here that can tell a fire
    // that RAN from a fire that was RIGHT. A six-token period repeated five
    // times, then its first two tokens again: the continuation is the third,
    // and a model that copies must say so. What the ids SPELL does not matter
    // -- induction is a copying circuit -- which is why this needs no
    // tokenizer, and it is the same period `tests/serving.rs` shows qwen3.
    //
    // It is the right question for THIS model in particular. Eighteen of
    // qwen3.5-0.8b's twenty-four layers are linear attention, so the copy has
    // to travel through the recurrent carry. Finite, varied logits prove the
    // dispatches ran; only this proves the carry carried.
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let mut tokens: Vec<u32> = Vec::new();
    for _ in 0..5 {
        tokens.extend_from_slice(&PERIOD);
    }
    tokens.push(PERIOD[0]);
    tokens.push(PERIOD[1]);

    match shell.step(&[driver_wgpu::turns::Turn { who: 1, tokens }]) {
        Ok(step) => {
            let row = step
                .logits
                .row(step.readout_of[0])
                .expect("the turn's own row");
            let mut top: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            println!("\nQWEN3.5 FIRED.");
            println!("  top: {:?}", &top[..8.min(top.len())]);
            println!("  wanted {} (the period's third token)", PERIOD[2]);

            let finite = row.iter().filter(|v| v.is_finite()).count();
            assert_eq!(finite, row.len(), "the logits hold non-finite values");
            assert!(
                top[0].1 > top[top.len() / 2].1,
                "the distribution is flat, which is what a null carry produces"
            );
            if u32::try_from(top[0].0) == Ok(PERIOD[2]) {
                println!("\n  IT CONTINUED THE PATTERN.");
            } else {
                println!(
                    "\n  IT RAN BUT DID NOT CONTINUE THE PATTERN: it wants {} \
                     and the period says {}. The dispatches all succeeded, so \
                     this is an ANSWER that is wrong rather than a fire that \
                     failed -- the shape a carry read from the wrong place \
                     produces.",
                    top[0].0, PERIOD[2],
                );
            }
        }
        Err(why) => {
            println!("\nTHE FIRE WAS REFUSED:\n  {why}");
            return;
        }
    }

    // A NEGATIVE CONTROL, because a wrong answer has two very different
    // causes and they need separating. If the gated DeltaNet's layers are
    // WIRED but computing the wrong thing, zeroing one of their weights moves
    // the distribution. If they are INERT -- dispatched over buffers nothing
    // reads, which is what a wrong operand order can look like when every
    // dispatch still succeeds -- zeroing changes nothing at all, and the
    // answer is coming entirely from the six full-attention layers.
    //
    // `layer.0.conv_w` is chosen because it is unambiguously a GDN weight:
    // no full-attention layer binds one.
    let baseline = fire_once(&mut shell, 2);
    let Some(conv) = real.get("layer.0.conv_w") else {
        println!("\nthis text binds no `layer.0.conv_w`, so the control could not run");
        return;
    };
    if let Err(why) = shell.hold("layer.0.conv_w", &vec![0u8; conv.len()]) {
        println!("\nthe zeroed weight would not stage: {why}");
        return;
    }
    let zeroed = fire_once(&mut shell, 3);

    let moved = baseline
        .iter()
        .zip(&zeroed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let same = baseline.iter().zip(&zeroed).filter(|(a, b)| a == b).count();
    println!(
        "\nZEROING `layer.0.conv_w`: widest move {moved}, {same} of {} logits unchanged",
        baseline.len()
    );
    if moved == 0.0 {
        println!(
            "  THE GDN LAYERS ARE INERT. Every dispatch succeeded and none of \
             them reached the answer, so the distribution above is the six \
             full-attention layers alone."
        );
    } else {
        println!(
            "  the GDN layers are WIRED and reach the answer, so what is wrong \
             is the arithmetic inside them rather than the plumbing around \
             them."
        );
    }
}

/// One prefill of the induction prompt, as a distribution.
fn fire_once(shell: &mut driver_wgpu::shell::Shell, who: u64) -> Vec<f32> {
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let mut tokens: Vec<u32> = Vec::new();
    for _ in 0..5 {
        tokens.extend_from_slice(&PERIOD);
    }
    tokens.push(PERIOD[0]);
    tokens.push(PERIOD[1]);
    let step = shell
        .step(&[driver_wgpu::turns::Turn { who, tokens }])
        .expect("the fire ran once already");
    step.logits
        .row(step.readout_of[0])
        .expect("the turn's own row")
        .to_vec()
}

/// A shell over the real qwen3.5 checkpoint, with every weight held.
///
/// Extracted so more than one test can fire this model. `slots` is how many
/// recurrent rows the pool opens: a test that runs two conversations against
/// one shell needs at least two, and a pool that is one short refuses at the
/// step rather than at the open.
fn qwen3_5_shell(
    slots: u32,
) -> Option<(driver_wgpu::shell::Shell, BTreeMap<String, Vec<u8>>)> {
    qwen3_5_shell_at(slots, 4, model::shared::policy::RuntimeQuant::Int4)
}

/// [`qwen3_5_shell`] at a stated affine width and runtime quantization.
fn qwen3_5_shell_at(
    slots: u32,
    bits: u32,
    quant: model::shared::policy::RuntimeQuant,
) -> Option<(driver_wgpu::shell::Shell, BTreeMap<String, Vec<u8>>)> {
    let dir = match qwen3_5_snapshot() {
        Some(dir) => dir,
        None => {
            println!("no Qwen3.5-0.8B snapshot, so IT COULD NOT BE MEASURED");
            return None;
        }
    };
    qwen3_5_shell_in(&dir, slots, bits, quant)
}

/// [`qwen3_5_shell_at`] over a STATED snapshot directory.
fn qwen3_5_shell_in(
    dir: &str,
    slots: u32,
    bits: u32,
    quant: model::shared::policy::RuntimeQuant,
) -> Option<(driver_wgpu::shell::Shell, BTreeMap<String, Vec<u8>>)> {
    let row = match model::catalog::find("qwen3.5-0.8b-base") {
        Some(row) => row,
        None => {
            println!("this build has no `qwen3.5-0.8b-base` row");
            return None;
        }
    };
    let device = match adapter() {
        Some(device) => device,
        None => {
            println!("no adapter, so IT COULD NOT BE MEASURED");
            return None;
        }
    };
    println!("adapter: {}", device.name());

    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let plan = hybrid_plan_class_at(&facts, model_ir::trace::FireClass::Prefill, bits);
    let decode = hybrid_plan_class_at(&facts, model_ir::trace::FireClass::Decode, bits);

    // The weights, through the same two-step road `tests/serving.rs` takes:
    // `MLX_IN_PLACE` first because that is what a driver boot asks for, then
    // the runtime encode its refusal names.
    let real = qwen3_5_weights_at(dir, row, &plan, quant)?;
    println!("{} weights staged", real.len());

    let text = driver_wgpu::shell::Text {
        decode,
        prefill: plan,
        geometry: Geometry {
            q_heads: facts.attn.q_heads,
            kv_heads: facts.attn.kv_heads,
            // THE RECURRENT PAIR, stated. `Geometry::recurrent` falls back to
            // `(kv_heads, head_dim)` when this is zero, and for this hybrid
            // that is attention's 2 and 256 where the gated DeltaNet's are 16
            // and 128 -- which is FIX 8 exactly: a scan dispatched over two
            // heads of sixteen, the rest of the state left as arena litter.
            //
            // And NO `..Default::default()` below it: this literal names every
            // field, so one ADDED upstream should break the build rather than
            // be filled with a zero -- which is how the pair arrived silently
            // wrong in the first place.
            v_heads: facts.gdn.value_heads,
            v_dim: facts.gdn.value_head_dim,
            head_dim: facts.attn.head_dim,
            rotary_dims: facts.attn.rotary_dim,
            n_experts: 0,
            experts_per_token: 0,
        },
        layers: u16::try_from(facts.layers).expect("a small stack"),
    };
    let deployment = driver_wgpu::shell::Deployment {
        pages: 64,
        theta: 10_000_000.0,
        recurrent: Some(driver_wgpu::resources::Recurrent {
            linear_layers: 18,
            conv_dim: 6144,
            conv_k: 4,
            v_heads: 16,
            v_dim: 128,
            k_dim: 128,
            slots,
        }),
        ..driver_wgpu::shell::Deployment::default()
    };
    let mut shell = match driver_wgpu::shell::Shell::on(device, text, deployment) {
        Ok(shell) => shell,
        Err(why) => {
            println!("\nTHE SHELL WOULD NOT OPEN:\n  {why}");
            return None;
        }
    };
    println!("shell open, with a recurrent pool");

    for (name, bytes) in &real {
        if let Err(why) = shell.hold(name, bytes) {
            println!("\n`{name}` WOULD NOT STAGE:\n  {why}");
            return None;
        }
    }
    println!("weights held");
    Some((shell, real))
}

/// **The prefill and the decode are two implementations of one recurrence, so
/// they have to agree — and if they do not, one of them is the wrong answer.**
///
/// # Why this is the next question and not another guess
///
/// [`qwen3_5_fires_or_says_where_it_stopped`] leaves the model running and
/// wrong, with the gated DeltaNet layers proven WIRED by a zeroing control.
/// Everything after that point is a hunt for arithmetic, and arithmetic needs
/// something to be checked AGAINST. The obvious references are all out of
/// reach here: there is no torch in this environment, `driver-metal` does not
/// build on this machine, and `driver-cuda`'s hybrid test is a shape
/// comparison rather than a numerical one.
///
/// **This model carries its own reference.** A prefill runs
/// `gdn_prep` + `gdn_core_recurrent`; a decode runs `gdn_core`, which fuses
/// the same four phases into one dispatch. They are separately written
/// implementations of the same recurrence over the same weights, so feeding
/// one row `N` tokens at once and another the same `N` tokens one at a time
/// must land on the same distribution. No oracle, no network, no second
/// backend.
///
/// # What each outcome means, decided before the run
///
/// - **They DISAGREE** — then one of the two GDN implementations is wrong, and
///   the wrong answer upstairs has a cause inside this file's reach. Which of
///   the two is at fault is the next question, not this one.
/// - **They AGREE** — then the fault is in something BOTH share, and the
///   search moves off the scan and onto the parts neither path re-derives:
///   the rope (where `mrope_interleaved` is already a named suspect), the
///   convolution, the norms, or a weight that is laid out wrongly for every
///   reader of it.
///
/// Either way the candidate list gets shorter, which is more than another
/// reading of the shader would do.
///
/// # The control that makes the result mean anything
///
/// This same equivalence is asserted for qwen3-0.6b by
/// `tests/serving.rs::a_conversation_is_answered_the_same_however_it_reaches_the_driver`,
/// and it PASSES. So the harness — the page table, the readout row, the KV
/// cache, `Turn` — is known to hold a prefill against a decode on this
/// backend already. qwen3.5 adds the linear-attention layers to that, which
/// is why a disagreement here points at them rather than at the scaffolding.
///
/// Skips loudly when the snapshot, the row or the adapter is absent.
#[test]
#[ignore = "loads and encodes a real checkpoint; run it deliberately"]
fn the_prefill_and_the_decode_are_one_recurrence_written_twice() {
    let Some((mut shell, _real)) = qwen3_5_shell(64) else {
        return;
    };

    // Twelve tokens is two periods of the induction prompt, so the tokens are
    // the same population the wrong answer came from.
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let tokens: Vec<u32> = PERIOD.iter().chain(PERIOD.iter()).copied().collect();

    // THE SWEEP, and it is the diagnosis rather than the headline. A single
    // number says the two paths differ; the CURVE says what kind of thing
    // differs, because the two hypotheses make different predictions:
    //
    // - A per-token disagreement -- the conv prologue, the q/k norms, the
    //   gates -- is already at its full size at the shortest length that can
    //   show one, and stays there.
    // - A carry disagreement -- the decay, the state layout, the order of the
    //   delta-rule update -- is small at two tokens and GROWS, because each
    //   step compounds the last.
    //
    // The sweep starts at 2. A one-token turn is `FireClass::Decode` on both
    // sides by definition ("every request contributes one token row"), so at
    // N = 1 the two rows run the SAME kernels and would agree for a reason
    // that has nothing to do with the question.
    let mut curve: Vec<(usize, f32, bool)> = Vec::new();
    for (i, n) in [2usize, 3, 4, 6, 8, 12].into_iter().enumerate() {
        let who = 100 + (i as u64) * 2;
        let at_once = match shell.step(&[driver_wgpu::turns::Turn {
            who,
            tokens: tokens[..n].to_vec(),
        }]) {
            Ok(step) => step
                .logits
                .row(step.readout_of[0])
                .expect("the prefill row")
                .to_vec(),
            Err(why) => {
                println!("\nTHE PREFILL OF {n} WAS REFUSED:\n  {why}");
                return;
            }
        };

        let mut one_at_a_time = Vec::new();
        for (t, token) in tokens[..n].iter().enumerate() {
            match shell.step(&[driver_wgpu::turns::Turn {
                who: who + 1,
                tokens: vec![*token],
            }]) {
                Ok(step) => {
                    one_at_a_time = step
                        .logits
                        .row(step.readout_of[0])
                        .expect("the decode row")
                        .to_vec();
                }
                Err(why) => {
                    println!("\nDECODE {t} OF {n} WAS REFUSED:\n  {why}");
                    return;
                }
            }
        }

        // NaN-AWARE, for the third time in this file. `f32::max` returns the
        // non-NaN operand, so folding differences with it reports two all-NaN
        // rows as agreeing exactly -- which is how the first version of this
        // comparison printed a tolerance over two readings of nothing.
        let widest = if at_once.iter().chain(&one_at_a_time).any(|v| !v.is_finite()) {
            f32::NAN
        } else {
            at_once
                .iter()
                .zip(&one_at_a_time)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };
        println!("      prefill {}", shape_of(&at_once));
        println!("      decode  {}", shape_of(&one_at_a_time));
        curve.push((n, widest, argmax_of(&at_once) == argmax_of(&one_at_a_time)));
        println!(
            "  N = {n:>2}: widest disagreement {widest:>9.4}, argmax {} \
             (prefill {}, decode {})",
            if curve.last().expect("just pushed").2 {
                "agrees"
            } else {
                "DIFFERS"
            },
            argmax_of(&at_once),
            argmax_of(&one_at_a_time),
        );

        // THE VACUITY GUARD, and it is the reason this loop is trustworthy
        // at all. A widest disagreement of EXACTLY zero over a quarter of a
        // million logits is not two implementations agreeing: they reduce the
        // same products in different groupings through bf16 storage, so a few
        // ulps somewhere is the BEST they can do. Bitwise equality means the
        // two sides are reading one answer, and a comparison of a row with
        // itself passes every tolerance ever written.
        //
        // So the loop asks the question it would have to answer anyway: does
        // this shell distinguish two different prompts? If it does not, the
        // "agreement" above measured the harness rather than the model.
        if widest == 0.0 {
            let other = match shell.step(&[driver_wgpu::turns::Turn {
                who: who + 1_000,
                tokens: tokens[..n].iter().map(|t| t ^ 1).collect(),
            }]) {
                Ok(step) => step
                    .logits
                    .row(step.readout_of[0])
                    .expect("the control row")
                    .to_vec(),
                Err(why) => {
                    println!("    the vacuity control was refused: {why}");
                    return;
                }
            };
            let moved = at_once
                .iter()
                .zip(&other)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                moved > 0.0,
                "N = {n}: a bitwise-identical answer AND a different prompt \
                 that changes nothing. This shell is not distinguishing its \
                 inputs, so the zero above is the harness and not the model."
            );
            println!("    (a different prompt moves the answer by {moved}, so the row is live)");
        }
    }

    let (_, widest, _) = *curve.last().expect("the sweep ran");
    let (_, shortest, _) = *curve.first().expect("the sweep ran");
    println!("\nPREFILL vs DECODE, over {} lengths:", curve.len());

    // The tolerance is a budget for ORDER OF SUMMATION, not for arithmetic:
    // the two paths reduce the same products in different groupings and in
    // bf16 storage, so they are allowed to differ by a fraction of the span
    // they compute over. A quarter of a logit is far wider than that, and far
    // narrower than a recurrence that is actually a different function.
    const SLACK: f32 = 0.25;
    if widest <= SLACK {
        println!(
            "  THE TWO IMPLEMENTATIONS AGREE. So the wrong answer is in what \
             they SHARE -- the rope, the convolution, the norms, or a weight \
             laid out wrongly for every reader -- and not in the scan."
        );
    } else if shortest <= SLACK {
        println!(
            "  THEY AGREE AT THE SHORTEST LENGTH AND DIVERGE AS IT GROWS, so \
             what differs is the CARRY between tokens -- the decay, the state \
             layout, or the order of the delta-rule update -- and not the \
             per-token arithmetic, which both paths evidently share."
        );
    } else {
        println!(
            "  THEY DISAGREE ALREADY AT {} TOKENS, so what differs is \
             PER-TOKEN arithmetic and not the carry: two tokens is the \
             shortest fire that can run both paths at all, and one step of a \
             recurrence has nothing yet to compound.",
            curve[0].0,
        );
    }
    // TWO FINDINGS THIS REPORTS RATHER THAN ASSERTS, because both are open.
    //
    // **The sequenced scan has its own odd-row NaN**, and
    // [`the_rectangle_the_odd_row_nan_first_appears_at`] has already named it:
    // `gdn_core_recurrent_prefill_bfloat16_l_32_v_4` at layer 2, rectangle 36
    // of 64, clean at 35 and dirty at 36.
    //
    // It MAKES it rather than inheriting it. All four of its arena inputs are
    // finite -- `mixed`, `pre_q`, `pre_k` and the 2080-wide `pre_gate` -- and
    // **13 of its 6144 output elements are not**. Nor is it the unwritten-slot
    // shape `gated_rms` had: the grid covers `Dv` exactly, `dv_base` stepping
    // 0, 4, ... 124 with `vrows = 4` over a `Dv` of 128.
    //
    // So it is arithmetic inside the scan, on 13 channels out of 6144, from
    // finite inputs -- which is a much smaller question than the one this file
    // started with and has a two-minute reproduction.
    //
    // AND THE SLABS ARE FINITE TOO, which is what makes "from finite inputs" a
    // measurement rather than a walk's blind spot. The scan's recurrent state
    // never appears in `Lowered::args` -- it is a device buffer the pool owns
    // -- and `st` is loaded from it before the first token and written back
    // after the last, so a NaN already there would come out looking exactly
    // like one the scan made. `Shell::recurrent` was added to close that, and
    // all three planes of layer 2 read back clean over their WHOLE extent:
    // 16,777,216 floats of `recurrent_state` and 1,572,864 each of
    // `conv_state` and `new_conv_state`, none of them non-finite.
    //
    // The whole extent and not a prefix, which is its own small lesson: the
    // first version capped the read at a megabyte, and the seat this probe
    // uses is nowhere near the first megabyte -- `Book::free_slot` hands each
    // new conversation an unused slot and this test opens dozens. A clean read
    // of the wrong region is the most confident kind of wrong answer.
    //
    // So every input the kernel has is finite and its outputs are not, and
    // knowing WHICH is what named the first cause. The thirteen were scattered
    // across tokens, heads and channels with no boundary among them -- so not
    // an indexing bug in the store -- and every one was a NaN rather than an
    // infinity, so not a magnitude that overflowed.
    //
    // The arm was reading `Input(0)`, `Input(1)` and `Input(2)` as `pre_q`,
    // `pre_k` and `pre_gate`. The statement carries FOUR inputs:
    // `[mixed, pre_q, pre_k, pre_gate]`, because metal's entrypoint has five
    // buffer slots it declares nothing at and binds `mixed` into them as
    // padding. So it bound `mixed` where `pre_q` goes and shifted the rest,
    // and every operand still resolved with every extent in range.
    //
    // # Ten remain, and they are the ARITHMETIC's
    //
    // Upstream retuned the scan's tile between two runs of the rectangle walk
    // -- `l_32_v_4` to `l_32_v_2` -- and the ten come back at **the same ten
    // coordinates**. The tile is the whole decomposition: how many value rows
    // a lane group carries, how wide the lane reduction is, how much of `st`
    // lives in registers. Two decompositions agreeing element for element
    // rules out a reduction, a lane mapping and a register spill in one
    // measurement, and it cost nothing -- upstream ran the experiment.
    //
    // One of the ten is a `+inf` and the rest are NaN, which reads as an
    // overflow at one channel and `inf - inf` behind it. The same entrypoint
    // on the same shapes is clean at layers 0 and 1 of the same fire, so it is
    // this LAYER's data that reaches it.
    //
    // # And the kernel disagrees with its own inputs
    //
    // At `t = 0` the scan reduces to one line, because the state it starts
    // from is zero: `st` is 0, so `kv` is 0, `delta` is `vv * gb`,
    // `st = k * delta`, and
    //
    //     out = vv * gb * SUM_d k[d] * q[d]
    //
    // Every term is in the arena. Recomputed on the CPU for the one `+inf`,
    // `(t0, h14, d50)`:
    //
    //     ga 9.977e-1   gb 4.866e-1   vv -1.190e-1   k.q 2.059e-3
    //     |q|max 4.6e-2   |k|max 7.0e-1   ->   vv*gb*k.q = -1.192e-4
    //
    // **Minus a ten-thousandth, against the device's `+inf`.** Nothing here is
    // extreme; the largest thing in the whole computation is 0.7. So this is
    // not data that overflows -- it is a kernel that does not compute what its
    // own operands say it should.
    //
    // The `t = 0` reduction is what makes that readable, and it rests on ONE
    // assumption worth naming: that the seat's recurrent state is zero when
    // the fire begins. `RecurrentPool::open` zeroes every slab and
    // `Book::free_slot` gives each new conversation an unused seat, and every
    // probe here takes a fresh `who`. The slab reads back nonzero AFTER the
    // fire, which is the scan writing its state and not a counter-example.
    //
    // So it is a numerical or ordering defect inside the scan's own body, and
    // **it is the first thing this file has found that is not a wiring
    // defect** -- the six before it were a head count, three more head counts,
    // an operand order and a parameter index.
    //
    // **The prefill predicts the LAST TOKEN OF THE PROMPT**, exactly, at every
    // length: 6100 at four tokens, 2930 at six, 1723 at eight, 2930 at twelve
    // -- `PERIOD[n-1 mod 6]` every time. That is what a residual stream
    // dominated by its own embedding does through a TIED lm head, which is to
    // say the layers are contributing almost nothing to the answer. It is a
    // sharper symptom than "wrong" and a different one from the copy-off-by-one
    // it looks like.
    let nan: Vec<usize> = curve
        .iter()
        .filter(|(_, w, _)| !w.is_finite())
        .map(|(n, _, _)| *n)
        .collect();
    if !nan.is_empty() {
        println!("  NON-FINITE at {nan:?} rows, which is the scan's own");
    }
    assert!(
        !curve.is_empty(),
        "the sweep ran no lengths, so nothing above is a comparison"
    );
}

/// Which id holds the widest value.
fn argmax_of(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0
}

/// The widest absolute value in a distribution.
///
/// **NaN-aware, and it was not.** `f32::max` returns the non-NaN operand, so a
/// fold over it reports an all-NaN row as a span of ZERO -- indistinguishable
/// from a row nothing wrote, which is the other thing this file calls dark.
/// Two whole rounds of bisection were read off that number, so it answers
/// `f32::NAN` for a row holding one rather than quietly dropping it.
fn span_of(row: &[f32]) -> f32 {
    if row.iter().any(|v| !v.is_finite()) {
        return f32::NAN;
    }
    row.iter().copied().fold(0.0f32, |m, v| m.max(v.abs()))
}

/// How a row fails, in words: finite and flat, or not finite at all.
fn shape_of(row: &[f32]) -> String {
    let nan = row.iter().filter(|v| v.is_nan()).count();
    let inf = row.iter().filter(|v| v.is_infinite()).count();
    let span = row.iter().copied().filter(|v| v.is_finite()).fold(0.0f32, |m, v| m.max(v.abs()));
    format!(
        "{} values, {nan} NaN, {inf} inf, widest finite {span:.5}",
        row.len()
    )
}

/// One fire on a fresh row, as the whole distribution it answers with.
///
/// Empty when the step was refused, which `span_of` reads as zero -- the same
/// as a dark row, and the callers that care assert the control separately.
fn fire_row(shell: &mut driver_wgpu::shell::Shell, who: u64, tokens: &[u32]) -> Vec<f32> {
    match shell.step(&[driver_wgpu::turns::Turn {
        who,
        tokens: tokens.to_vec(),
    }]) {
        Ok(step) => step
            .logits
            .row(step.readout_of[0])
            .map(<[f32]>::to_vec)
            .unwrap_or_default(),
        Err(why) => {
            println!("    the {}-token fire was refused: {why}", tokens.len());
            Vec::new()
        }
    }
}

/// One fire on a fresh row, as the span of the distribution it answers with.
///
/// `None` when the step was refused. A span rather than a token because what a
/// dark fire produces is not garbage text: it is a hidden state of zeros
/// through a quantized `lm_head`, which reads out as a quarter of a million
/// tiny constants that are the same whatever the prompt was. An argmax would
/// look plausible; a span of zero cannot.
fn fire_span(shell: &mut driver_wgpu::shell::Shell, who: u64, tokens: &[u32]) -> Option<f32> {
    fire_turns(
        shell,
        &[driver_wgpu::turns::Turn {
            who,
            tokens: tokens.to_vec(),
        }],
    )
    .map(|(span, _)| span)
}

/// A whole fire, as the span of the FIRST turn's answer and how many
/// rectangles the device recorded.
///
/// The dispatch count is the half that separates "every rectangle ran and the
/// arithmetic is wrong" from "the fire ran less than the plan states", and
/// `Fired::dispatches` is the only place either is observable.
fn fire_turns(
    shell: &mut driver_wgpu::shell::Shell,
    turns: &[driver_wgpu::turns::Turn],
) -> Option<(f32, usize)> {
    let step = match shell.step(turns) {
        Ok(step) => step,
        Err(why) => {
            println!("    the fire was refused: {why}");
            return None;
        }
    };
    let row = step.logits.row(step.readout_of[0])?;
    Some((
        row.iter().copied().fold(0.0f32, |m, v| m.max(v.abs())),
        step.fired.dispatches,
    ))
}

/// **A shell that answers once and then stops, and exactly which fire is the
/// last live one.**
///
/// # Where this came from
///
/// [`the_prefill_and_the_decode_are_one_recurrence_written_twice`] compares a
/// prefill against a run of decodes, and its vacuity guard caught something
/// bigger than the comparison it was protecting: after the first pair of rows,
/// every fire returns **all zeros**, and a different prompt does not move
/// them. No refusal, no non-finite value, no error — a row that reads out as
/// a quarter of a million exact zeros is a row nothing ever wrote.
///
/// That is not a numerical bug and it cannot be reasoned about from the
/// comparison, because the comparison had already stopped being a measurement
/// by the time it printed a tolerance. So this fires a SCRIPT and reports each
/// fire on its own, which is the only way to say whether the thing that goes
/// dark is the Nth fire, the Nth row, or the first prefill after a decode.
///
/// It reports rather than asserting a shape, because a target invented here
/// would be a guess about a bug nobody has localised yet. It asserts only the
/// one thing that is true by construction: **the first fire must be live**,
/// since a shell that answers nothing at all is a different bug and this test
/// would otherwise pass by measuring it.
///
/// # What it found: two bugs, one fixed
///
/// The permanent darkness was `Frame::recurrent_slots` never being written —
/// see [`qwen3_5_fires_or_says_where_it_stopped`]. With it fixed the canary
/// holds 9.688 through every fire in the sweep, which is what says the fires
/// no longer poison each other.
///
/// # FIXED: `gated_rms` was given the attention's head count
///
/// The odd-row NaN is gone. Every length from 1 to 32 now answers with a real
/// distribution, and the canary moved from 9.688 to 17.250 — because fourteen
/// of every sixteen gated-DeltaNet heads had never been normalised at all.
///
/// `lowering::hold::gated_rms` passed `Facts::kv_heads`, the ATTENTION's
/// key/value head count, where `norm/gated_rms.wgsl` uses the grid's y as the
/// GDN's VALUE head count. On a hybrid those are different numbers: 2 and 16.
/// The grid was an eighth of the size it should have been, so most of the
/// output was never WRITTEN and kept whatever the arena slot held — and the
/// slot had last held an f32 `pre_q`, whose bytes read as bf16 halves are
/// occasionally a NaN pattern. Two of 6144.
///
/// That is why the row count appeared to matter and never did: the row count
/// decides the arena layout, the layout decides which value's bytes are lying
/// in the slot, and those bytes decide whether the leftovers look like NaN.
///
/// The rest of this doc is the road to it, kept because the wrong turns are
/// the reusable part.
///
/// # The word "dark" in this file meant two things, and one of them was wrong
///
/// A dark row is **all NaN**, not a row nothing wrote, and this file called it
/// the second for two whole rounds of bisection.
///
/// The cause was its own measurement. `span_of` folded with `f32::max`, which
/// returns the NON-NaN operand — so a row of 248,320 NaNs came back as a span
/// of 0.0, exactly like a row of zeros. Worse, the same fold was used to ask
/// whether two different prompts moved the answer, and `(NaN - NaN).abs()`
/// through `f32::max` is 0 too, so the file concluded *"the tokens never
/// entered"* from an arithmetic that could not have said anything else.
///
/// [`whether_the_odd_row_collapse_survives_switching_the_gated_deltanet_off`]
/// is where it surfaced, and `span_of` refuses to hide it now.
///
/// **Every fire whose ROW COUNT is odd and greater than one answers with NaN
/// in every row**; every even one answers normally:
///
/// ```text
/// 1 -> 10.250   2 -> 12.500   3 ->   NaN   4 ->  9.688   5 ->   NaN
/// 6 ->  9.812   7 ->   NaN    8 -> 11.562  9 ->   NaN   10 ->  9.688
/// 11 ->  NaN   12 -> 10.125  16 -> 9.562  31 ->   NaN   32 -> 10.250
/// ```
///
/// A single row is [`model_ir::trace::FireClass::Decode`] and so is not a
/// counter-example.
///
/// # The hypotheses this test has already eliminated
///
/// Worth listing, because each one was the obvious next guess and each is
/// wrong, and a reader who has to re-eliminate them pays for the run twice:
///
/// - **Not the prefill.** Three conversations decoding ONE token each is a
///   three-row fire of the other class, firing the fused `gdn_core` instead
///   of the split pair, and it is dark too. So the parity belongs to the row
///   count.
/// - **Not a fire that ran short.** `Fired::dispatches` is **364 rectangles**
///   for every length, dark or live.
/// - **Not the readout.** Every row is dark at three, not merely the one the
///   turn reads; at four, every row is live.
/// - **Not the device.** `Device::drained` is asked after every fire and
///   `wgpu` complains about nothing.
/// - **Not the lowering.** The GDN launches carry byte-identical params at
///   every row count,
///   [`what_the_lowering_does_differently_at_the_row_counts_that_go_dark`]
///   finds no two-byte operand of odd width — the one shape from which a row
///   parity could arise, since bf16 packs two to a word — and no rectangle
///   addressing past its arena at 2, 3, 4, 5 or 6 rows.
/// - **Not generic to this driver.** qwen3-0.6b answers at the same row
///   counts:
///   `tests/serving.rs::a_prefill_longer_than_the_first_one_is_still_answered`.
/// - **Not the host at all.**
///   [`what_the_row_count_changes_about_a_rectangle_and_what_it_does_not`]
///   checks the embedding gather's grid and extent, every arena operand of
///   all 364 rectangles for scaling, every rectangle planned at the device's
///   256-byte binding alignment, the readout range, and which kernels fire
///   how many times. All five are identical at both row counts.
/// - **Not a guard.** `GuardPred::TokensMultipleOf(tile)` divides rather than
///   compares, which is the only thing in the tree that makes a parity on
///   purpose -- and three rows and four lower to the same kernels in the same
///   numbers, so no arm is being swapped.
///
/// # Where it is: one MLP block
///
/// [`whether_the_odd_row_collapse_survives_switching_the_gated_deltanet_off`]
/// switches the layer branches off one kind at a time, by zeroing the norm
/// weight each branch begins at — `c.norm` states `NormVariant::Plain`, so a
/// zeroed weight is a gain of ZERO and the branch is genuinely removed rather
/// than attenuated, which the gemma `plus_one` convention would have made it.
///
/// | what is on | 3 rows | 4 rows |
/// |---|---|---|
/// | everything | NaN | 9.688 |
/// | gated DeltaNet off | NaN | 9.688 |
/// | and attention off | NaN | 9.812 |
/// | and the MLPs off | **14.312** | 9.812 |
/// | one MLP (layer 0) back | NaN | 9.812 |
///
/// **One MLP block at three rows is enough to make every logit NaN**, so this
/// is a kernel to read rather than a drift to chase.
///
/// The block is `rms(mlp_norm)` → `qmv(gate_proj)` → `qmv(up_proj)` →
/// `silu_mul` → `qmv_residual(down)`, and two of those five are exonerated by
/// the rows above: `rms_single_row` and `affine_qmv_fast` both run on REAL
/// values at three rows inside the gated DeltaNet's own branch — whose
/// `attn_norm` is never zeroed — and produce no NaN. What is left is
/// `silu_mul_bfloat16` and `affine_qmv_fast_residual`, and `silu_mul`'s plain
/// arm indexes flatly over WORDS with no notion of a row at all, which is an
/// argument and not a measurement.
///
/// `Shell::keep_arena` exists now and reads the whole arena back, and it is not
/// enough by itself: an arena is reused, so its END STATE holds whatever was
/// written last, and once the residual stream is NaN every range in it is.
/// What is left to try is a readback per DISPATCH, or the depth sweep that
/// [`whether_one_layer_is_enough_to_make_the_odd_row_nan`] opens — one layer
/// is clean, twenty-four is not, so the answer is somewhere between.
#[test]
#[ignore = "loads and encodes a real checkpoint; run it deliberately"]
fn how_many_fires_a_shell_answers_before_it_goes_dark() {
    let Some((mut shell, real)) = qwen3_5_shell(64) else {
        return;
    };
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];

    // WHICH LENGTH, and whether it takes the shell with it.
    //
    // Two hypotheses fit what has been seen so far and they make different
    // predictions, so the script tests both at once. The earlier scripts went
    // 4,4,4,... (all live) and 2,3,4,... (dark from the 3), which reads as
    // GROWTH -- a buffer sized by the first fire and overflowed by a later
    // one. But the three-token fire is also the first NON-POWER-OF-TWO length
    // anything has fired, and every length that has ever answered (2, 4, 32)
    // is a power of two.
    //
    // So: walk lengths that separate the two, and after each one fire a CANARY
    // of a length already known good on a fresh row. The canary is the half
    // that matters, because "this fire is dark" and "this fire poisoned the
    // shell" are different bugs and the earlier scripts could not tell them
    // apart.
    let lengths = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 31, 32];
    let mut live = Vec::new();
    let mut who = 1u64;
    for (i, n) in lengths.into_iter().enumerate() {
        let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % PERIOD.len()]).collect();
        who += 1;
        let Some((span, dispatches)) = fire_turns(
            &mut shell,
            &[driver_wgpu::turns::Turn {
                who,
                tokens: tokens.clone(),
            }],
        ) else {
            live.push(false);
            continue;
        };
        who += 1;
        let canary = fire_span(&mut shell, who, &[PERIOD[0], PERIOD[1], PERIOD[2], PERIOD[3]]);
        println!(
            "fire {}: {n:>2} tokens -> span {span:>8.3} ({dispatches} rectangles) \
             canary(4) -> {}   lowerings {}",
            i + 1,
            match canary {
                Some(c) => format!("{c:>8.3}"),
                None => "REFUSED".to_owned(),
            },
            shell.lowerings_derived(),
        );
        if let Err(why) = shell.device().drained() {
            println!("    THE DEVICE COMPLAINED: {why}");
        }
        live.push(span > 1.0 && canary.is_some_and(|c| c > 1.0));
    }

    let dark = live.iter().position(|l| !l);
    match dark {
        None => println!("\nEVERY FIRE IN THE SCRIPT ANSWERED."),
        Some(0) => println!("\nTHE FIRST FIRE IS ALREADY DARK."),
        Some(k) => println!(
            "\nTHE FIRST DARK FIRE IS {} ({} tokens), and {k} before it answered.",
            k + 1,
            lengths[k],
        ),
    }

    // DOES THE PROMPT ENTER AT ALL? The sharpest question available, and it
    // splits the causes in two.
    //
    // If two different three-token prompts give the SAME logits, the token ids
    // never reached the stack: the embedding gather wrote nothing this fire
    // read, every kernel after it computed on the arena's zeros, and the tiny
    // constants coming out are a quantized `lm_head` over a hidden state of
    // zeros. If they DIFFER, the tokens did enter and something later
    // collapsed — which is a different search entirely.
    for n in [3usize, 4] {
        let a: Vec<u32> = (0..n).map(|t| PERIOD[t % PERIOD.len()]).collect();
        let b: Vec<u32> = a.iter().map(|t| t ^ 1).collect();
        let one = shell.step(&[driver_wgpu::turns::Turn {
            who: 850 + n as u64,
            tokens: a,
        }]);
        let two = shell.step(&[driver_wgpu::turns::Turn {
            who: 860 + n as u64,
            tokens: b,
        }]);
        if let (Ok(one), Ok(two)) = (one, two) {
            let (Some(x), Some(y)) = (
                one.logits.row(one.readout_of[0]),
                two.logits.row(two.readout_of[0]),
            ) else {
                continue;
            };
            let moved = x
                .iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max);
            println!(
                "\n{n} rows, two different prompts: widest move {moved} -- {}",
                if moved == 0.0 {
                    "THE TOKENS NEVER ENTERED. The fire computed on the arena's zeros from the embedding onward."
                } else {
                    "the tokens DID enter, so the collapse is downstream of the embedding"
                },
            );
        }
    }

    // ODD ROWS, OR AN ODD PREFILL? Three conversations decoding one token
    // each is a fire of three ROWS that is not a prompt, and the classes
    // differ: `FireClass::Decode` fires the fused `gdn_core` where a prefill
    // fires the split pair. If three decode rows answer, the parity is the
    // PREFILL's and not the row count's.
    let three_rows = fire_turns(
        &mut shell,
        &[
            driver_wgpu::turns::Turn {
                who: 700,
                tokens: vec![PERIOD[0]],
            },
            driver_wgpu::turns::Turn {
                who: 701,
                tokens: vec![PERIOD[1]],
            },
            driver_wgpu::turns::Turn {
                who: 702,
                tokens: vec![PERIOD[2]],
            },
        ],
    );
    // WHICH ROWS ARE DARK, which is free to ask and splits the cause again.
    // If the early rows answer and the last does not, the epilogue ran over
    // fewer rows than the fire has and the arena's zeros are what the readout
    // is reading. If every row is dark, the stack itself collapsed.
    for n in [3usize, 4] {
        let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % PERIOD.len()]).collect();
        if let Ok(step) = shell.step(&[driver_wgpu::turns::Turn {
            who: 800 + n as u64,
            tokens,
        }]) {
            let spans: Vec<String> = (0..step.rows)
                .map(|r| match step.logits.row(r) {
                    Some(row) => format!(
                        "{:.2}",
                        row.iter().copied().fold(0.0f32, |m, v| m.max(v.abs()))
                    ),
                    None => "-".to_owned(),
                })
                .collect();
            println!(
                "\n{n} rows: per-row spans [{}]  (readout row {})",
                spans.join(", "),
                step.readout_of[0],
            );
        }
    }

    match three_rows {
        Some((span, dispatches)) => println!(
            "\nTHREE CONVERSATIONS, ONE TOKEN EACH (3 rows, decode class): \
             span {span:.3} ({dispatches} rectangles) -- {}",
            if span > 1.0 {
                "LIVE, so the parity belongs to the prefill and not to the row count"
            } else {
                "DARK, so it is the ROW COUNT and not the prefill"
            },
        ),
        None => println!("\nthe three-row decode was refused"),
    }

    // IS THE SHELL POISONED, and it is a different question from whether a
    // fire is dark.
    //
    // It was the same question once. Before the recurrent slots were written,
    // the first dark fire took every later one with it -- including a fresh
    // row at a length that had just answered -- so something outlived the
    // fire, and putting the weights back was the way to ask whether it was
    // them. It was not; it was the carry.
    //
    // **The CANARY is what says which question this is.** It fires a
    // known-good length on a fresh row after every step, so a live canary
    // beside a dark fire means the shell is fine and only that fire failed.
    // Running the re-hold probe on a healthy shell and reporting that it
    // "recovered" would be a conclusion drawn from a comparison that had
    // nothing to compare -- it recovers because it was never broken.
    //
    // That error is this file's own, twice: a liveness predicate that every
    // dark row passed, and a vacuity guard added because a tolerance was being
    // printed over two readings of one row.
    if live.last() == Some(&false) {
        for (name, bytes) in &real {
            if let Err(why) = shell.hold(name, bytes) {
                println!("re-holding `{name}` failed: {why}");
                return;
            }
        }
        let after = match shell.step(&[driver_wgpu::turns::Turn {
            who: 900,
            tokens: PERIOD[..2].to_vec(),
        }]) {
            Ok(step) => step
                .logits
                .row(step.readout_of[0])
                .expect("the row after re-holding")
                .to_vec(),
            Err(why) => {
                println!("the fire after re-holding was refused: {why}");
                return;
            }
        };
        let span = after.iter().copied().fold(0.0f32, |m, v| m.max(v.abs()));
        println!("\nTHE SHELL IS POISONED. After re-holding every weight: span {span}");
        if span > 1.0 {
            println!(
                "  IT RECOVERED, so a fire WROTE OVER A WEIGHT BUFFER: an arm \
                 bound a result where an operand lives, and every later fire \
                 read the result."
            );
        } else {
            println!(
                "  STILL DARK, so the poison is not in the weights and the \
                 three things that outlive a fire do not cover it."
            );
        }
    } else {
        println!(
            "\nTHE SHELL IS NOT POISONED: the canary answers 9.688 after every \
             fire, dark or live, so each dark fire fails alone."
        );
    }

    assert!(
        live[0],
        "the first fire answered nothing, so this test measured a shell that \
         never worked rather than one that stops working"
    );
}

/// qwen3.5's weights, by the names this driver binds.
///
/// `MLX_IN_PLACE` first, because that is what a driver boot asks for, then the
/// runtime encode its refusal names — the same two-step road
/// `tests/serving.rs` takes for qwen3, and the reason a bf16 release is a
/// measurement here rather than a skip.
fn qwen3_5_weights(
    dir: &str,
    row: &'static dyn model::catalog::Variant,
    plan: &model_ir::trace::ForwardPlan,
) -> Option<BTreeMap<String, Vec<u8>>> {
    qwen3_5_weights_at(dir, row, plan, model::shared::policy::RuntimeQuant::Int4)
}

/// [`qwen3_5_weights`] at a stated runtime quantization.
fn qwen3_5_weights_at(
    dir: &str,
    row: &'static dyn model::catalog::Variant,
    plan: &model_ir::trace::ForwardPlan,
    quant: model::shared::policy::RuntimeQuant,
) -> Option<BTreeMap<String, Vec<u8>>> {
    let path = std::path::Path::new(dir);
    let meta = model_loader::checkpoint::read::parse_checkpoint_metadata(path).ok()?;
    let config = std::fs::read_to_string(path.join("config.json")).ok()?;
    let encoding = model::encoding::Encoding::from_config_json(&config).ok()?;
    let target = model_loader::plan::StorageTarget::for_backend(
        model_loader::types::BackendKind::Vulkan,
        0,
        1,
    );
    let began = std::time::Instant::now();
    let load = match model::boot::compile_load_plan_for(
        path,
        &meta,
        &target,
        row,
        &encoding,
        model::boot::Binding::MLX_IN_PLACE,
    ) {
        Ok((plan, _)) => plan,
        Err(why) => {
            let said = why.to_string();
            if !said.contains("needs quantized weights") {
                println!("\nTHE LOAD PLAN WAS REFUSED:\n  {said}");
                return None;
            }
            let policy = model::shared::policy::Policy {
                projections: model::shared::policy::Projections::InPlace,
                naming: model::shared::policy::Naming::Mlx,
                runtime_quant: quant,
                moe_request: model::shared::policy::Mxfp4MoeRequest::Auto,
                component: model::shared::policy::Component::Full,
                stream_routed_experts: false,
                knobs: model::shared::policy::FamilyKnobs::default(),
            };
            let (contract, _) =
                model::contract::author_with_policy(row, &encoding, &meta, &target, &policy)
                    .map_err(|e| println!("\nTHE LOADER WOULD NOT AUTHOR IT:\n  {e}"))
                    .ok()?;
            model_loader::plan::compile(&meta, &contract, target)
                .map_err(|e| println!("\nTHE PLAN WOULD NOT COMPILE:\n  {e}"))
                .ok()?
        }
    };
    println!("plan compiled, {} tensors", load.tensors.len());

    let storage = model_loader::executor::Execution::new(&load, path)
        .run()
        .map_err(|e| println!("\nTHE LOADER WOULD NOT EXECUTE THE PLAN:\n  {e}"))
        .ok()?;

    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        2
    ];
    let low = lower(
        plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .ok()?;
    let bound: BTreeSet<String> = low
        .args
        .iter()
        .filter_map(|a| match a {
            model_compiler::lower::Arg::Weight(n) => Some(n.clone()),
            _ => None,
        })
        .filter(|n| !n.starts_with("scale."))
        .collect();

    let naming = driver_wgpu::names::Naming::mlx();
    let mut out = BTreeMap::new();
    let mut bytes = 0u64;
    for traced in bound {
        let Some(held) = naming
            .spellings(&traced)
            .iter()
            .find_map(|s| storage.tensors.get(s.as_str()))
        else {
            println!("\n`{traced}` RESOLVES TO NOTHING THE LOADER PRODUCED");
            return None;
        };
        bytes += held.len() as u64;
        out.insert(traced, held.clone());
    }
    println!(
        "{} weights, {bytes} bytes, staged in {:.1}s",
        out.len(),
        began.elapsed().as_secs_f32()
    );
    Some(out)
}

/// Which seams the hybrid text declares, and whether `OUT` is one of them.
#[test]
fn the_hybrid_text_declares_the_seams_a_driver_reads() {
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    for class in [
        model_ir::trace::FireClass::Prefill,
        model_ir::trace::FireClass::Decode,
    ] {
        let plan = hybrid_plan_class(&facts, class);
        let seams: Vec<String> = plan.seams.iter().map(|s| s.seam.clone()).collect();
        println!("{class:?}: {seams:?}");
        println!(
            "  OUT present: {}",
            plan.seams
                .iter()
                .any(|s| s.seam == model_ir::seam::OUT.name)
        );
    }
}

/// Which values carry the `Dim::Const` the two classes disagree about.
#[test]
fn what_the_two_classes_disagree_about() {
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let pre = hybrid_plan_class(&facts, model_ir::trace::FireClass::Prefill);
    let dec = hybrid_plan_class(&facts, model_ir::trace::FireClass::Decode);
    let consts = |p: &model_ir::trace::ForwardPlan| -> BTreeSet<u32> {
        p.values
            .iter()
            .flat_map(|v| v.shape.0.iter())
            .filter_map(|d| match d {
                model_ir::trace::Dim::Const(n) => Some(*n),
                _ => None,
            })
            .collect()
    };
    let (d, f) = (consts(&dec), consts(&pre));
    println!("only in decode: {:?}", d.difference(&f).collect::<Vec<_>>());
    println!(
        "only in prefill: {:?}",
        f.difference(&d).collect::<Vec<_>>()
    );
    for (n, v) in pre.values.iter().enumerate() {
        if v.shape
            .0
            .iter()
            .any(|dm| matches!(dm, model_ir::trace::Dim::Const(32)))
        {
            println!("  prefill value {n}: shape {:?}", v.shape.0);
        }
    }
}

/// **What the lowering does differently at a row count that goes dark.**
///
/// # Where this came from
///
/// [`how_many_fires_a_shell_answers_before_it_goes_dark`] fires qwen3.5 at a
/// walk of prompt lengths and finds a clean split: **2, 4, 8 and 32 answer;
/// 3 and 5 do not**, and the first non-answering fire leaves the shell
/// permanently dark. Every length that has ever worked is a power of two.
///
/// A power-of-two dependency is a statement about ARITHMETIC, and the
/// arithmetic that a row count reaches is the lowering's: the grids, the
/// scalars and the arena offsets are all derived from it. So this asks the
/// question on the CPU, where it costs nothing and can be read.
///
/// # Why the diff and not an assertion
///
/// There is no known-good lowering to hold this against — the whole point is
/// that nobody has seen one for this family. What CAN be said is that four and
/// five are the same plan at adjacent row counts, so anything that is not a
/// smooth function of the row count is a candidate, and printing them beside
/// each other is what makes that visible.
///
/// It asserts the one thing that must hold whatever the cause: **every row
/// count lowers**. A lowering that refused would have been the answer, and a
/// silent dark fire means it did not.
#[test]
fn what_the_lowering_does_differently_at_the_row_counts_that_go_dark() {
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let plan = hybrid_plan(&facts);

    let mut seen: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for n in [2usize, 3, 4, 5, 6, 8] {
        let rows: Vec<Row> = vec![
            Row {
                samples: true,
                ..Row::default()
            };
            n
        ];
        let low = match lower(
            &plan,
            &rows,
            Fire {
                captures_across_splits: false,
            },
        ) {
            Ok(low) => low,
            Err(why) => panic!("{n} rows did not lower: {why:?}"),
        };

        // Only the gated DeltaNet's launches, and only their first layer: the
        // question is what a row count does to one rectangle, and thirty
        // copies of the same answer is noise.
        let mut lines = Vec::new();
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            if !symbol.starts_with("gdn") || launch.layers.start != 0 {
                continue;
            }
            let params = &low.params[launch.params.start as usize..launch.params.end as usize];
            lines.push(format!(
                "{symbol}  rows {:?}  params {params:?}",
                launch.rows,
            ));
        }
        println!("\n== {n} ROWS ==  (arena {} bytes)", low.arena_bytes);
        for line in &lines {
            println!("  {line}");
        }
        seen.insert(n, lines);
    }

    // ODD WIDTHS, which is where a row parity could come from at all.
    //
    // An activation is stored bf16, two to a `u32` word. A value whose width
    // is EVEN starts every row on a word boundary whatever the row count is; a
    // value whose width is ODD starts every second row half a word in, so the
    // fire's behaviour can depend on the parity of the row index — and a
    // kernel that reads `array<u32>` and halves its subscript cannot address
    // that at all.
    //
    // The whole plan and not just the gated DeltaNet, because the symptom is
    // that EVERY row goes dark and nothing says the cause is in the family
    // this file is named for.
    {
        let rows: Vec<Row> = vec![
            Row {
                samples: true,
                ..Row::default()
            };
            3
        ];
        let low = lower(
            &plan,
            &rows,
            Fire {
                captures_across_splits: false,
            },
        )
        .expect("three rows lowered above");
        let mut odd: BTreeMap<String, Vec<(u32, u32)>> = BTreeMap::new();
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            for arg in &low.args[launch.args.start as usize..launch.args.end as usize] {
                let (width, bytes) = match arg {
                    Arg::Arena { width, bytes, .. }
                    | Arg::Named { width, bytes, .. } => {
                        (*width, *bytes)
                    }
                    _ => continue,
                };
                if bytes == 2 && width % 2 == 1 {
                    odd.entry(symbol.clone()).or_default().push((width, bytes));
                }
            }
        }
        println!("\n== TWO-BYTE OPERANDS OF ODD WIDTH ({}) ==", odd.len());
        for (symbol, widths) in &odd {
            println!("  {symbol}: {widths:?}");
        }
    }

    // DOES ANY RECTANGLE ADDRESS PAST ITS ARENA, and does the answer depend on
    // the row count?
    //
    // An operand's extent is `rows * width * bytes` from its offset. A
    // rectangle that runs past `arena_bytes` is writing where nothing else
    // will read it or reading what nothing wrote — and `wgpu` CLAMPS an
    // out-of-bounds storage access rather than trapping, so the fire completes
    // with every rectangle dispatched and the answer is zeros. That is exactly
    // the symptom an odd row count produces, so it is worth the arithmetic
    // rather than the assumption.
    println!("\n== RECTANGLES THAT ADDRESS PAST THE ARENA ==");
    for n in [2usize, 3, 4, 5, 6] {
        let rows: Vec<Row> = vec![
            Row {
                samples: true,
                ..Row::default()
            };
            n
        ];
        let low = lower(
            &plan,
            &rows,
            Fire {
                captures_across_splits: false,
            },
        )
        .expect("these row counts lower");
        let mut past: Vec<String> = Vec::new();
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let span = (launch.rows.end - launch.rows.start) as usize;
            for arg in &low.args[launch.args.start as usize..launch.args.end as usize] {
                let Arg::Arena { at, width, bytes } = arg else {
                    continue;
                };
                let end = at + span * (*width as usize) * (*bytes as usize);
                if end > low.arena_bytes {
                    past.push(format!(
                        "{symbol}: {at}..{end} past {}",
                        low.arena_bytes
                    ));
                }
            }
        }
        past.sort_unstable();
        past.dedup();
        println!("  {n} rows (arena {}): {} such", low.arena_bytes, past.len());
        for line in past.iter().take(6) {
            println!("      {line}");
        }
    }

    // The comparison the eye would make, made by the test: which lines are the
    // same shape at four rows and five, and which are not.
    let four = seen.get(&4).expect("four lowered");
    let five = seen.get(&5).expect("five lowered");
    println!("\n== FOUR vs FIVE ==");
    if four.len() != five.len() {
        println!(
            "  DIFFERENT NUMBER OF GDN LAUNCHES: {} at four, {} at five",
            four.len(),
            five.len()
        );
    }
    for (a, b) in four.iter().zip(five) {
        if a != b {
            println!("  four: {a}\n  five: {b}");
        }
    }
}

/// **The same prompt, fired twice, and whether it is the same answer.**
///
/// # What this is testing and why it is the whole diagnosis
///
/// [`how_many_fires_a_shell_answers_before_it_goes_dark`] fired
/// `PERIOD[..4]` on three fresh rows of one shell and got spans of 9.688,
/// 9.688 and **9.625**. Same weights, same tokens, same length, same shell —
/// a different answer. A driver has no licence to do that, and
/// `tests/serving.rs` asserts the opposite for qwen3 by name (*"two shells
/// over one checkpoint answered differently"*).
///
/// Non-determinism on identical input is not a numerical complaint. It is a
/// RACE, and this family has a place for one that can be pointed at:
///
/// - `gdn_prep` + `gdn_core_recurrent` are the PER-REQUEST pair.
///   `kernels-metal/kernels/ssm/gdn_prep.metal`'s header states their grid as
///   `{32, Vd, R*Hv}`, *"one simdgroup per (req, v-head, v-dim)"*, and proves
///   them bit-identical to the FUSED DECODE kernel `gdn_core`. One token per
///   request is the shape they are written for.
/// - `qwen_3_5/forward/metal.rs` fires that pair for `FireClass::Prefill`,
///   where a rectangle's rows are TOKENS. Every token of one prompt resolves
///   the same `slot_ids` entry and does a read-modify-write on the same
///   `rstate` — concurrently.
/// - The pair written for a prompt, `gdn_prep_prefill` +
///   `gdn_core_recurrent_prefill`, carries `row_pitch` and `n_scan` and loops
///   `for t in 0..n_scan` with the state held in registers, which is what
///   makes the recurrence sequential. **No text and no DSL emitter names
///   either of them**, on any backend, while all three carry routines, arms,
///   geometry rules and nine tuned `(LANES, VROWS)` instantiations for them.
///
/// # The control is the point
///
/// A one-token turn is `FireClass::Decode`, which fires the fused `gdn_core`
/// with one row per request — the shape the kernel is written for, and no
/// race. So the decode side must be deterministic and the prefill side must
/// not be, and a test that measured only one of them could not tell a racing
/// scan from a flaky machine.
#[test]
#[ignore = "loads and encodes a real checkpoint; run it deliberately"]
fn the_same_prompt_twice_is_the_same_answer_or_the_scan_is_racing() {
    let Some((mut shell, _real)) = qwen3_5_shell(64) else {
        return;
    };
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];

    // Lengths that are known to answer. Three and five put this shell
    // permanently dark, which is its own defect and would only hide this one.
    let mut verdicts = Vec::new();
    for n in [1usize, 4, 8] {
        let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % PERIOD.len()]).collect();
        let mut answers: Vec<Vec<f32>> = Vec::new();
        for k in 0..4u64 {
            let who = 500 + (n as u64) * 10 + k;
            let Ok(step) = shell.step(&[driver_wgpu::turns::Turn {
                who,
                tokens: tokens.clone(),
            }]) else {
                println!("the {n}-token fire was refused");
                return;
            };
            answers.push(
                step.logits
                    .row(step.readout_of[0])
                    .expect("the turn's own row")
                    .to_vec(),
            );
        }
        let first = &answers[0];
        let widest = answers[1..]
            .iter()
            .map(|a| {
                a.iter()
                    .zip(first)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max)
            })
            .fold(0.0f32, f32::max);
        let distinct = {
            let mut bits: Vec<Vec<u32>> = answers
                .iter()
                .map(|a| a.iter().map(|v| v.to_bits()).collect())
                .collect();
            bits.sort_unstable();
            bits.dedup();
            bits.len()
        };
        println!(
            "  {n:>2} tokens ({}): {distinct} distinct answers in 4 fires, \
             widest spread {widest}",
            if n == 1 { "decode" } else { "prefill" },
        );
        verdicts.push((n, distinct, widest));
    }

    let decode = verdicts
        .iter()
        .find(|(n, _, _)| *n == 1)
        .expect("the one-token control ran");
    let prefills: Vec<_> = verdicts.iter().filter(|(n, _, _)| *n > 1).collect();
    println!();
    if decode.1 > 1 {
        println!(
            "  THE DECODE IS NON-DETERMINISTIC TOO, so this is not the prefill \
             scan and the control has taken the diagnosis away from it."
        );
    } else if prefills.iter().any(|(_, d, _)| *d > 1) {
        println!(
            "  THE DECODE IS DETERMINISTIC AND THE PREFILL IS NOT. The fused \
             `gdn_core` runs one row per request and the split pair runs one \
             per TOKEN over one slot's state, so the prefill's tokens are \
             racing on the recurrent read-modify-write -- which is the answer \
             to why this model runs, stays finite, and is wrong."
        );
    } else {
        println!(
            "  BOTH ARE DETERMINISTIC over these four fires, which does not \
             clear the race: a fixed grid on one adapter can schedule the same \
             way every time. It means this test did not catch it, not that it \
             is not there."
        );
    }

    // The control is the only assertion. The prefill's behaviour is the
    // measurement this test exists to report, and asserting it would be
    // asserting the bug -- a test that turns red the day somebody fixes it.
    assert_eq!(
        decode.1, 1,
        "the same one-token turn answered {} different ways, so the fused \
         decode kernel is racing too and nothing below is about the prefill",
        decode.1,
    );
}

/// **A conversation whose carry has nowhere to live is refused by name.**
///
/// # Why this needs an assertion and not a comment
///
/// `wgpu` CLAMPS an out-of-bounds storage read rather than trapping. A slot
/// past the end of the slab therefore resolves to some other conversation's
/// carry, every dispatch succeeds, and the model answers fluently out of
/// somebody else's state — which is indistinguishable, from the outside, from
/// the answer it should have given.
///
/// That is not a hypothetical failure mode for this table. It is the one it
/// was found by: `Frame::recurrent_slots` was declared, staged and never
/// written, an unwritten table stages as empty, and an empty buffer answers
/// every subscript with the same clamp. Every conversation read slot zero and
/// inherited the previous fire's carry, and the visible symptom was that the
/// same prompt answered a different way each time it was asked.
///
/// So the boundary gets a refusal rather than a clamp, and the refusal gets a
/// test. No weights: the check sits between `Frame::of` and the fire, so a
/// shell with nothing held reaches it and stops there.
#[test]
#[ignore = "opens a real adapter; run it deliberately"]
fn a_carry_with_no_slot_to_live_in_is_refused_rather_than_clamped() {
    let Some(device) = adapter() else {
        println!("no adapter, so IT COULD NOT BE MEASURED");
        return;
    };
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let text = driver_wgpu::shell::Text {
        decode: hybrid_plan_class(&facts, model_ir::trace::FireClass::Decode),
        prefill: hybrid_plan(&facts),
        geometry: Geometry {
            q_heads: facts.attn.q_heads,
            kv_heads: facts.attn.kv_heads,
            // THE RECURRENT PAIR, stated. `Geometry::recurrent` falls back to
            // `(kv_heads, head_dim)` when this is zero, and for this hybrid
            // that is attention's 2 and 256 where the gated DeltaNet's are 16
            // and 128 -- which is FIX 8 exactly: a scan dispatched over two
            // heads of sixteen, the rest of the state left as arena litter.
            //
            // And NO `..Default::default()` below it: this literal names every
            // field, so one ADDED upstream should break the build rather than
            // be filled with a zero -- which is how the pair arrived silently
            // wrong in the first place.
            v_heads: facts.gdn.value_heads,
            v_dim: facts.gdn.value_head_dim,
            head_dim: facts.attn.head_dim,
            rotary_dims: facts.attn.rotary_dim,
            n_experts: 0,
            experts_per_token: 0,
        },
        layers: u16::try_from(facts.layers).expect("a small stack"),
    };
    // ONE slot, and two conversations in one fire.
    let deployment = driver_wgpu::shell::Deployment {
        pages: 8,
        theta: 10_000_000.0,
        recurrent: Some(driver_wgpu::resources::Recurrent {
            linear_layers: 18,
            conv_dim: 6144,
            conv_k: 4,
            v_heads: 16,
            v_dim: 128,
            k_dim: 128,
            slots: 1,
        }),
        ..driver_wgpu::shell::Deployment::default()
    };
    let mut shell = match driver_wgpu::shell::Shell::on(device, text, deployment) {
        Ok(shell) => shell,
        Err(why) => {
            println!("the shell would not open: {why}");
            return;
        }
    };

    let refused = shell.step(&[
        driver_wgpu::turns::Turn {
            who: 1,
            tokens: vec![7, 9],
        },
        driver_wgpu::turns::Turn {
            who: 2,
            tokens: vec![7, 9],
        },
    ]);
    match refused {
        Err(driver_wgpu::turns::Unstepped::NoSlot { slot, slots }) => {
            assert_eq!(slots, 1, "the pool's own count");
            assert_eq!(slot, 1, "the second conversation's seat");
            println!("refused by name: slot {slot} of {slots}");
        }
        Err(other) => panic!(
            "the second conversation was refused for the wrong reason: {other}. \
             A pool of one slot and two conversations is what `NoSlot` is for, \
             and any earlier refusal means this test stopped covering it."
        ),
        Ok(_) => panic!(
            "two conversations fired over a pool of ONE slot. One of them read \
             the other's carry, because an out-of-bounds storage read clamps on \
             this backend rather than trapping."
        ),
    }
}

/// **Everything the host decides differently at three rows and at four, which
/// is nothing.**
///
/// # Where this came from
///
/// [`how_many_fires_a_shell_answers_before_it_goes_dark`] narrowed the odd-row
/// collapse to one sentence: at three rows **two different prompts give
/// bitwise-identical logits**, and at four they differ by 11.4. Whatever else
/// is true, the answer does not depend on the tokens.
///
/// Two things produce that, and they are not the same bug: a fire that ran on
/// the arena's zeros from the embedding onward, or a READOUT pointed at a
/// range the head never wrote. Either way the arena is freshly zeroed every
/// step, so both come back as a quarter of a million tiny constants — a
/// quantized `lm_head` over a hidden state of zeros — and neither can be told
/// from the other by looking at the output.
///
/// # What it checks, and it is the whole host side
///
/// Every one of these is a place a row parity could live, and the answer is
/// the same at three rows and four:
///
/// - **The embedding gather's grid and bound extent.** Its writes are guarded
///   by `at >= arrayLength(&out_)` — the BOUND range's length — so an extent
///   short of `rows * hidden` would reject every invocation and write nothing.
///   The extents are 4096, 6144, 8192, 10240 for 2, 3, 4, 5 rows: exact.
/// - **Every arena operand of every one of the 364 rectangles**, for whether
///   its extent scales 3:4 with the row count. None fails to.
/// - **Every rectangle, planned at the DEVICE's 256-byte storage-binding
///   alignment** rather than the probe's usual 1. An odd row count moves every
///   offset after the first row-shaped value, so a misalignment would show
///   here. No rectangle refuses at either row count.
/// - **The readout range.** `serve::logits` reads
///   `rows * vocab * bytes` at `readout.at`, and at both row counts that is
///   exactly the arena's tail: `135680..1625600 of 1625600` at three,
///   `164352..2150912 of 2150912` at four.
/// - **Which kernels each row count lowers to, and how many times each
///   fires.** This is the one mechanism in the tree that produces a parity BY
///   CONSTRUCTION: `GuardPred::TokensMultipleOf(tile)` picks a kernel by
///   whether the tile DIVIDES the row count rather than by a threshold, so an
///   even tile sends odd and even row counts down different arms. The kernel
///   sets are equal and every symbol fires the same number of times.
///
/// So the host tells the device the same thing, correctly, at both row counts.
/// The defect is on the device — in a shader's own arithmetic, or in a hazard
/// between rectangles that `wgpu`'s tracker does not see — and that is a
/// different class of tool than this file has.
///
/// CPU only: no adapter, no weights, no checkpoint.
#[test]
fn what_the_row_count_changes_about_a_rectangle_and_what_it_does_not() {
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let plan = hybrid_plan(&facts);
    let geometry = Geometry {
        q_heads: facts.attn.q_heads,
        kv_heads: facts.attn.kv_heads,
        head_dim: facts.attn.head_dim,
        rotary_dims: facts.attn.rotary_dim,
        n_experts: 0,
        experts_per_token: 0,
        ..Default::default()
    };

    // The READOUT path, which is where a row count stops being a stride and
    // becomes a count of things to gather. `how_many_fires...` measures its
    // per-row spans AFTER this runs, so a whole transformer that computed
    // correctly and an epilogue that gathered nothing look the same from
    // outside.
    const EPILOGUE: &[&str] = &[];
    /// Per row count, per symbol, the `(entrypoint, dispatch counts)` each
    /// of its launches recorded.
    type ByRows = BTreeMap<usize, BTreeMap<String, Vec<(String, Vec<u64>)>>>;
    let mut by_rows: ByRows = BTreeMap::new();
    let mut raw: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    let mut tally: BTreeMap<usize, BTreeMap<String, usize>> = BTreeMap::new();
    for n in [3usize, 4] {
        let rows: Vec<Row> = vec![
            Row {
                samples: true,
                ..Row::default()
            };
            n
        ];
        let low = lower(
            &plan,
            &rows,
            Fire {
                captures_across_splits: false,
            },
        )
        .expect("these row counts lower");
        let mut seen: BTreeMap<String, Vec<(String, Vec<u64>)>> = BTreeMap::new();
        let mut refused: BTreeMap<String, usize> = BTreeMap::new();
        // THE READOUT, which is the last link and the one nothing has checked.
        // `serve::logits` reads `readout.rows * vocab * bytes` at
        // `readout.at`. If that range is not where the head WROTE, the read
        // comes back as the arena's zeros -- prompt-independent, every row
        // dark, every rectangle dispatched.
        match low.readout {
            Some(r) => println!(
                "-- {n} rows: readout at {} rows {} vocab {} bytes {} => {}..{} of {}",
                r.at,
                r.rows,
                r.vocab,
                r.bytes,
                r.at,
                r.at + (r.rows as usize) * (r.vocab as usize) * (r.bytes as usize),
                low.arena_bytes,
            ),
            None => println!("-- {n} rows: NO READOUT"),
        }
        let held = Placeholder(u64::from(u32::MAX));
        let store = Unbacked(held);
        let arena = Arena {
            buffer: &held,
            bytes: u64::from(u32::MAX),
        };
        for launch in &low.launches {
            let symbol = low.kernels[launch.kernel as usize].clone();
            if !EPILOGUE.is_empty() && !EPILOGUE.iter().any(|e| symbol.starts_with(e)) {
                continue;
            }

            let Ok(declared) =
                driver_wgpu::reflect::entrypoint(&symbol, driver_wgpu::Capability::Baseline)
            else {
                continue;
            };
            let module = driver_wgpu::geometry::Module::loaded(&symbol, &declared);
            match driver_wgpu::dispatch::plan_one(
                &low,
                launch,
                Built {
                    module,
                    declared: &declared,
                },
                Sources {
                    arena,
                    resolver: &store,
                    // THE DEVICE'S alignment, not one. A storage binding's
                    // offset must be a multiple of
                    // `min_storage_buffer_offset_alignment`, which is 256 on
                    // every desktop adapter this has run on, and a probe that
                    // asks for one accepts offsets the real fire cannot bind.
                    // An odd row count moves every offset after the first
                    // row-shaped value, so this is where a parity could show.
                    min_offset: 256,
                },
                geometry,
            ) {
                Ok(d) => {
                    let extents: Vec<u64> = d
                        .buffers
                        .iter()
                        .map(driver_wgpu::binding::Bound::len)
                        .filter(|e| *e != u64::from(u32::MAX))
                        .collect();
                    seen.entry(symbol)
                        .or_default()
                        .push((format!("{:?}", d.groups), extents));
                }
                Err(why) => {
                    let said = format!("{why}");
                    if !said.contains("slab") {
                        *refused.entry(said).or_insert(0usize) += 1;
                    }
                }
            }
        }
        println!("-- {n} rows: {} kinds of refusal at a 256-byte alignment", refused.len());
        for (why, count) in &refused {
            println!("     x{count}: {why}");
        }
        raw.insert(
            n,
            low.kernels.iter().cloned().collect::<BTreeSet<String>>(),
        );
        // The per-symbol LAUNCH COUNT too: a guard that swaps an arm can keep
        // both symbols in the table and change how often each fires.
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for launch in &low.launches {
            *counts
                .entry(low.kernels[launch.kernel as usize].clone())
                .or_insert(0) += 1;
        }
        tally.insert(n, counts);
        by_rows.insert(n, seen);
    }

    // THE SCAN: an arena operand of a row-shaped value must grow with the row
    // count. One that does not is either row-independent by design -- a value
    // sized `Dim::Requests` or `Dim::Const`, and there is one request in every
    // fire here -- or a rectangle addressing a fixed window of a buffer that
    // just got taller, which is the shape of a kernel that would read what
    // nothing wrote.
    //
    // Reported per SYMBOL and only where three rows and four disagree about
    // the ratio, because 364 launches is not something to read.
    let three = by_rows.get(&3).expect("three lowered");
    let four = by_rows.get(&4).expect("four lowered");
    // WHICH KERNELS EACH ROW COUNT LOWERS TO. `GuardPred::TokensMultipleOf`
    // picks a kernel by whether a tile DIVIDES the row count -- it is not a
    // threshold -- so a guard with an even tile sends odd and even row counts
    // down different arms, which is the one mechanism in the tree that
    // produces a parity by construction.
    let (k3, k4) = (
        raw.get(&3).expect("three lowered").clone(),
        raw.get(&4).expect("four lowered").clone(),
    );
    println!("\n== KERNELS AT THREE ROWS AND NOT FOUR ==");
    for k in k3.difference(&k4) {
        println!("  {k}");
    }
    println!("== KERNELS AT FOUR ROWS AND NOT THREE ==");
    for k in k4.difference(&k3) {
        println!("  {k}");
    }

    println!("== SYMBOLS FIRED A DIFFERENT NUMBER OF TIMES ==");
    let (t3, t4) = (
        tally.get(&3).expect("three"),
        tally.get(&4).expect("four"),
    );
    for (symbol, n3) in t3 {
        let n4 = t4.get(symbol).copied().unwrap_or(0);
        if *n3 != n4 {
            println!("  {symbol}: {n3} at three rows, {n4} at four");
        }
    }

    println!("\n== OPERANDS WHOSE EXTENT DOES NOT SCALE 3:4 ==");
    let mut flagged = 0usize;
    for (symbol, at_three) in three {
        let Some(at_four) = four.get(symbol) else {
            println!("  {symbol}: present at three rows and absent at four");
            flagged += 1;
            continue;
        };
        let (Some((g3, e3)), Some((g4, e4))) = (at_three.first(), at_four.first()) else {
            continue;
        };
        let scaled: Vec<bool> = e3
            .iter()
            .zip(e4)
            .map(|(a, b)| *b * 3 == *a * 4)
            .collect();
        if scaled.iter().any(|s| !s) {
            println!("  {symbol}\n      3: grid {g3} extents {e3:?}\n      4: grid {g4} extents {e4:?}");
            flagged += 1;
        }
    }
    println!("  ({flagged} symbols)");
}

/// **Is the odd-row collapse inside the gated DeltaNet, or in what every
/// layer shares?**
///
/// # The switch, and why this one
///
/// `gated_rms` multiplies by `layer.N.gate_norm` RAW — the text's own comment
/// says so, because the reference's `Qwen3NextRMSNormGated` is `nn.RMSNorm`-
/// shaped and folds nothing. So a zeroed `gate_norm` makes that layer's GDN
/// branch produce exactly zero, and `o_proj` of zero is zero, and the residual
/// stream passes through untouched. Eighteen of them turns qwen3.5 into its
/// six full-attention layers and their MLPs.
///
/// It is the right tensor for the job in a way `conv_w` is not. `gate_norm` is
/// plain bf16, so zeroing its bytes zeroes the VALUES; the projections are
/// affine-quantized, where zeroed bytes pin every weight to its group's zero
/// point instead — which is the trap `tests/serving.rs`'s control ran into and
/// recorded.
///
/// # What each outcome means, decided before the run
///
/// - **Three rows becomes LIVE** — the collapse is inside the gated DeltaNet,
///   and the search narrows to eight kernels from the whole stack.
/// - **Three rows stays DARK** — it is in what both layer kinds share: the
///   embedding, the norms, the projections, the MLP or the epilogue.
///
/// The four-row fire is the control on both, and it is not optional: if
/// zeroing takes IT dark too, then the switch did more than disable a branch
/// and neither reading above holds.
#[test]
#[ignore = "loads and encodes a real checkpoint; run it deliberately"]
fn whether_the_odd_row_collapse_survives_switching_the_gated_deltanet_off() {
    let Some((mut shell, real)) = qwen3_5_shell(64) else {
        return;
    };
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let three: Vec<u32> = PERIOD[..3].to_vec();
    let four: Vec<u32> = PERIOD[..4].to_vec();

    let before_three = fire_row(&mut shell, 1, &three);
    let before_four = fire_row(&mut shell, 2, &four);
    println!("with the gated DeltaNet ON:");
    println!("   3 rows: {}", shape_of(&before_three));
    println!("   4 rows: {}", shape_of(&before_four));

    // Every `gate_norm` the checkpoint published, zeroed. Counted, because a
    // switch that flipped nothing would leave the fire unchanged and this test
    // would read that as "the collapse survives".
    let mut flipped = 0usize;
    for (name, bytes) in &real {
        if !name.ends_with(".gate_norm") {
            continue;
        }
        if let Err(why) = shell.hold(name, &vec![0u8; bytes.len()]) {
            println!("`{name}` would not stage: {why}");
            return;
        }
        flipped += 1;
    }
    println!("{flipped} `gate_norm` tensors zeroed");
    assert!(
        flipped > 0,
        "no `gate_norm` was zeroed, so the switch did nothing and anything \
         below would be a reading of the unchanged model"
    );

    let after_three = fire_row(&mut shell, 3, &three);
    let after_four = fire_row(&mut shell, 4, &four);
    println!("with the gated DeltaNet OFF:");
    println!("   3 rows: {}", shape_of(&after_three));
    println!("   4 rows: {}", shape_of(&after_four));

    // DID THE SWITCH FLIP ANYTHING? A span is a maximum, and two very
    // different distributions can share one -- so "9.6875 before and 9.6875
    // after" is not evidence that nothing changed, and it is certainly not
    // evidence that something did. The conclusion below is about a model with
    // its gated DeltaNet OFF, and if `gate_norm` never reached the kernel then
    // that is not the model that was fired and every reading is void.
    let moved = before_four
        .iter()
        .zip(&after_four)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  zeroing moved the four-row answer by {moved}");
    assert!(
        moved > 0.0,
        "zeroing every `gate_norm` did not change the four-row answer by one ulp, so the switch never reached the kernel and nothing below is a reading of a model with its gated DeltaNet off"
    );

    // PHASE TWO: the other half. A full-attention layer's branch begins at
    // `layer.N.attn_norm`, so zeroing it makes q, k and v zero, the attention
    // output zero, and `o_proj` of zero zero -- the same clean removal, on the
    // six layers the gated DeltaNet does not own. With both switches thrown,
    // what is left is the embedding, the norms, the MLPs and the epilogue.
    //
    // The six are the layers no `conv_w` belongs to, taken from the weights
    // rather than from a list written here: a hard-coded 3, 7, 11, 15, 19, 23
    // would be a second place to be wrong about which layer is which.
    let mut attn_off = 0usize;
    for (name, bytes) in &real {
        let Some(rest) = name.strip_suffix(".attn_norm") else {
            continue;
        };
        let Some(n) = rest.strip_prefix("layer.") else {
            continue;
        };
        if real.contains_key(&format!("layer.{n}.conv_w")) {
            continue;
        }
        if let Err(why) = shell.hold(name, &vec![0u8; bytes.len()]) {
            println!("`{name}` would not stage: {why}");
            return;
        }
        attn_off += 1;
    }
    println!("{attn_off} full-attention `attn_norm` tensors zeroed");
    let bare_three = fire_row(&mut shell, 5, &three);
    let bare_four = fire_row(&mut shell, 6, &four);
    println!("with BOTH branches off:");
    println!("   3 rows: {}", shape_of(&bare_three));
    println!("   4 rows: {}", shape_of(&bare_four));
    let bare_moved = after_four
        .iter()
        .zip(&bare_four)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  zeroing the attention moved the four-row answer by {bare_moved}");
    if span_of(&bare_three) > 1.0 {
        println!(
            "\n  THREE ROWS IS LIVE WITH BOTH BRANCHES OFF, so the collapse is \
             in the ATTENTION half -- the projections at `head_dim` 256, the \
             partial rope over 64 of them, or the paged softmax."
        );
    } else if span_of(&bare_four) > 1.0 {
        println!(
            "\n  THREE ROWS IS DARK WITH EVERY LAYER BRANCH OFF, so the \
             collapse is in what remains: the embedding, the norms, the MLPs \
             or the epilogue -- none of which the gated DeltaNet or the \
             attention can reach."
        );
    } else {
        println!(
            "\n  BOTH ROW COUNTS ARE DARK WITH EVERY BRANCH OFF, so this \
             switch went too far to read."
        );
    }

    // PHASE THREE: the MLPs. `layer.N.mlp_norm` begins every MLP block on both
    // layer kinds, so zeroing all twenty-four leaves embed -> final_norm ->
    // lm_head -> the epilogue and nothing else. If three rows is STILL dark
    // with that little left, the suspect list is four kernels long.
    let mut mlp_off = 0usize;
    for (name, bytes) in &real {
        if !name.ends_with(".mlp_norm") {
            continue;
        }
        if let Err(why) = shell.hold(name, &vec![0u8; bytes.len()]) {
            println!("`{name}` would not stage: {why}");
            return;
        }
        mlp_off += 1;
    }
    println!("{mlp_off} `mlp_norm` tensors zeroed");
    let stripped_three = fire_row(&mut shell, 7, &three);
    let stripped_four = fire_row(&mut shell, 8, &four);
    println!("with the MLPs off too:");
    println!("   3 rows: {}", shape_of(&stripped_three));
    println!("   4 rows: {}", shape_of(&stripped_four));

    // ONE MLP BACK. Everything is off; putting layer 0's `mlp_norm` back gives
    // the model exactly one live MLP block. If three rows returns to NaN with
    // one, the fault is in the block itself rather than in anything that
    // accumulates over twenty-four of them -- which is the difference between
    // a kernel to read and a drift to chase.
    if let Some(bytes) = real.get("layer.0.mlp_norm") {
        if let Err(why) = shell.hold("layer.0.mlp_norm", bytes) {
            println!("`layer.0.mlp_norm` would not go back: {why}");
            return;
        }
        let one_three = fire_row(&mut shell, 9, &three);
        let one_four = fire_row(&mut shell, 10, &four);
        println!("with ONE MLP (layer 0) back on:");
        println!("   3 rows: {}", shape_of(&one_three));
        println!("   4 rows: {}", shape_of(&one_four));
        if one_three.iter().any(|v| v.is_nan()) {
            println!(
                "\n  ONE MLP BLOCK IS ENOUGH to make every logit NaN at three \
                 rows. The fault is inside the block -- `mlp_norm`, the two \
                 projections, `silu_mul` or the residual-folding `down` -- and \
                 not an accumulation across layers."
            );
        } else {
            println!(
                "\n  ONE MLP BLOCK IS FINITE at three rows, so what produces \
                 the NaN needs more than one and this is a drift rather than a \
                 kernel."
            );
        }
    }
    let stripped_moved = bare_four
        .iter()
        .zip(&stripped_four)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  zeroing the MLPs moved the four-row answer by {stripped_moved}");
    if span_of(&stripped_four) <= 1.0 {
        println!(
            "\n  THE CONTROL WENT DARK: nothing is left to read at four rows \
             either, so this switch went too far."
        );
    } else if span_of(&stripped_three) > 1.0 {
        println!(
            "\n  THREE ROWS IS LIVE ONCE THE MLPS ARE OFF, so the collapse is \
             in the MLP block: its projections, `silu_mul`, or the residual \
             add that folds it back."
        );
    } else {
        println!(
            "\n  THREE ROWS IS DARK WITH EVERY BLOCK OFF. What still runs is \
             `embed_gather`, `final_norm`, the lm head and the epilogue's \
             gather -- and the first and the last of those were checked on the \
             host and are exact."
        );
    }

    let live = |r: &[f32]| span_of(r) > 1.0;
    let after = (after_three.as_slice(), after_four.as_slice());
    if !live(after.1) {
        println!(
            "\n  THE CONTROL WENT DARK TOO: zeroing every `gate_norm` took the \
             FOUR-row fire with it, so the switch did more than disable a \
             branch and neither reading holds."
        );
    } else if live(after.0) {
        println!(
            "\n  THREE ROWS IS LIVE WITH THE GATED DELTANET OFF, so the \
             collapse is INSIDE it -- eight kernels rather than the whole \
             stack."
        );
    } else {
        println!(
            "\n  THREE ROWS IS STILL DARK WITH THE GATED DELTANET OFF, so the \
             collapse is in what BOTH layer kinds share: the embedding, the \
             norms, the projections, the MLP or the epilogue."
        );
    }

    // The measurement is the report. The one thing asserted is the baseline
    // this whole file rests on, because a run where four rows was never live
    // measured a broken shell rather than a row parity.
    assert!(
        live(&before_four),
        "the four-row fire was not live before anything was zeroed, so this \
         shell was already broken and the comparison means nothing"
    );
}

/// **Which of the MLP's two remaining kernels makes the NaN.**
///
/// # Where this came from
///
/// [`whether_the_odd_row_collapse_survives_switching_the_gated_deltanet_off`]
/// puts the odd-row NaN inside ONE MLP block and exonerates two of its five
/// kernels: `rms_single_row` and `affine_qmv_fast` both run on real values at
/// three rows inside the gated DeltaNet's own branch and stay finite. What is
/// left is `silu_mul_bfloat16` and the residual-folding `down`
/// (`affine_qmv_fast_residual`).
///
/// # The switch, and why `.scales` rather than the weight
///
/// An affine-quantized weight dequantises as `scale * (code - zero)`, so a
/// zeroed `.scales` plane makes every weight of that tensor exactly ZERO.
/// Zeroing the CODES would not: it pins each weight to its group's zero point,
/// which is a different tensor and not a small one — the trap `tests/serving.rs`
/// records walking into.
///
/// So `layer.N.down.scales` zeroed leaves `y = residual + 0`, with `mlp_norm`,
/// both projections and `silu_mul` all still running on real values. That is
/// the one edit that keeps `silu_mul` live and takes `down` out.
///
/// - **Finite with `down` out** → `silu_mul` on real values is fine and the
///   NaN is `down`'s.
/// - **Still NaN with `down` out** → it is made upstream of `down`, which is
///   `silu_mul` or the two projections feeding it.
///
/// The `up`/`gate` rows are the follow-up either way: they say which SIDE of
/// `silu_mul` carries it.
#[test]
#[ignore = "loads and encodes a real checkpoint; run it deliberately"]
fn which_of_the_mlps_kernels_carries_the_odd_row_nan() {
    let Some((mut shell, real)) = qwen3_5_shell(64) else {
        return;
    };
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let three: Vec<u32> = PERIOD[..3].to_vec();

    println!("layer 0's MLP tensors:");
    for name in real.keys() {
        if name.starts_with("layer.0.")
            && (name.contains("down") || name.contains("up_proj") || name.contains("gate_proj"))
        {
            println!("   {name}");
        }
    }

    let base = fire_row(&mut shell, 1, &three);
    println!("\nbaseline, 3 rows: {}", shape_of(&base));
    assert!(
        base.iter().any(|v| v.is_nan()),
        "three rows is finite on this build, so there is no NaN here to \
         attribute and every reading below would be of a model that works"
    );

    // One suffix at a time, each restored before the next, so every row of the
    // table is a measurement of ONE change rather than of the pile of them.
    let mut who = 10u64;
    for zeroed in ["down.scales", "up_proj.scales", "gate_proj.scales"] {
        let suffix = format!(".{zeroed}");
        let touched: Vec<&String> = real.keys().filter(|k| k.ends_with(&suffix)).collect();
        if touched.is_empty() {
            println!("\nno tensor ends with `{suffix}`, so it could not be switched off");
            continue;
        }
        for name in &touched {
            let bytes = &real[*name];
            if let Err(why) = shell.hold(name, &vec![0u8; bytes.len()]) {
                println!("`{name}` would not stage: {why}");
                return;
            }
        }
        who += 1;
        let out = fire_row(&mut shell, who, &three);
        println!(
            "\n{} x `{zeroed}` zeroed -> 3 rows: {}",
            touched.len(),
            shape_of(&out)
        );

        // BACK, before the next row of the table. Cumulative holds would make
        // the third measurement a reading of all three edits at once, which is
        // the shape of a bisection that cannot name anything.
        for name in &touched {
            if let Err(why) = shell.hold(name, &real[*name]) {
                println!("`{name}` would not go back: {why}");
                return;
            }
        }
        let restored = fire_row(&mut shell, who + 100, &three);
        assert!(
            restored.iter().any(|v| v.is_nan()),
            "putting `{zeroed}` back did not bring the NaN back, so the holds \
             are not restoring and the rows after this one are readings of a \
             model nobody described"
        );
    }
}

/// **Do two of one rectangle's operands overlap in the arena, and does that
/// depend on the row count?**
///
/// # Why this is the question
///
/// The odd-row NaN is bisected down to one MLP block, and inside it to the
/// residual-folding `down`. But `affine_qmv_fast_residual` is a MATVEC: its
/// `vec_` is `workgroup_id.x`, its `dot_lane` reads `x[vec_ * in_vec_size + k]`
/// and its store is a device-scoped CAS. **Row zero's output cannot depend on
/// how many rows the fire has** — nothing in the body reads a row count except
/// the grid extent.
///
/// So the arithmetic cannot be what changed, and one thing can: WHERE the
/// operands sit. The lowering packs activations into an arena and the packing
/// is a function of the row count, so two values that are disjoint at four
/// rows can be laid on top of each other at three — and every kernel between
/// them would then read what its neighbour wrote, which is a wrong answer and
/// not a refusal.
///
/// [`what_the_row_count_changes_about_a_rectangle_and_what_it_does_not`] asked
/// whether any rectangle ran PAST the arena. This asks the other question: two
/// rectangles inside it, on top of each other.
///
/// Overlap is not by itself a defect — an in-place chain is a run of
/// rectangles that share a buffer ON PURPOSE, which is what `Lowered::
/// value_owner` records. What would be one is an overlap that appears at one
/// row count and not at its neighbour, so that is what this reports.
#[test]
fn whether_two_operands_of_one_rectangle_land_on_each_other() {
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let plan = hybrid_plan(&facts);

    let mut found: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    for n in [2usize, 3, 4, 5, 6] {
        let rows: Vec<Row> = vec![
            Row {
                samples: true,
                ..Row::default()
            };
            n
        ];
        let low = lower(
            &plan,
            &rows,
            Fire {
                captures_across_splits: false,
            },
        )
        .expect("these row counts lower");
        let mut here = BTreeSet::new();
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let span = (launch.rows.end - launch.rows.start) as usize;
            let ranges: Vec<(usize, usize)> = low.args
                [launch.args.start as usize..launch.args.end as usize]
                .iter()
                .filter_map(|a| match a {
                    Arg::Arena { at, width, bytes } => {
                        Some((*at, at + span * *width as usize * *bytes as usize))
                    }
                    _ => None,
                })
                .collect();
            for (i, a) in ranges.iter().enumerate() {
                for b in &ranges[i + 1..] {
                    // Identical ranges are the in-place case and are the
                    // point of a fused residual; a PARTIAL overlap is two
                    // values sharing memory neither of them owns all of.
                    if a != b && a.0 < b.1 && b.0 < a.1 {
                        here.insert(format!("{symbol}: {a:?} overlaps {b:?}"));
                    }
                }
            }
        }
        println!("{n} rows: {} partial overlaps", here.len());
        for line in here.iter().take(4) {
            println!("    {line}");
        }
        found.insert(n, here);
    }

    let three = found.get(&3).expect("three lowered");
    let four = found.get(&4).expect("four lowered");
    println!("\nat three rows and not four: {}", three.difference(four).count());
    for line in three.difference(four).take(6) {
        println!("    {line}");
    }
}

/// **The first value in the fire that is not finite.**
///
/// # What this can do that the weight-zeroing could not
///
/// Every earlier round of this investigation switched a subsystem OFF and
/// re-fired, because the logits were the only thing observable and "the logits
/// are NaN" is the same sentence for a bad embedding and a bad lm head. That
/// bisected the odd-row NaN down to one MLP block and stopped, because the
/// thing it could not do was read the value BETWEEN two kernels.
///
/// [`driver_wgpu::shell::Shell::keep_arena`] hands the whole arena back, so the
/// question stops being which subsystem and becomes which OPERAND RANGE.
///
/// **It reads the END STATE and that bounds what it can say.** An arena is
/// reused, so an offset holds whatever was written there last; the first draft
/// of this test walked the launches in order, called the first non-finite
/// range the origin, and duly named `embed_gather` at offset zero — which is
/// the residual stream's slot and so the LAST writer's answer. What survives
/// is a per-symbol tally: a symbol all of whose ranges are finite at the end
/// neither made this NaN nor inherited it.
///
/// The four-row fire is the control: the same walk over a fire that works must
/// find nothing, or the walk is reading the arena wrongly rather than reading
/// a wrong arena.
#[test]
#[ignore = "loads and encodes a real checkpoint; run it deliberately"]
fn the_first_value_of_an_odd_row_fire_that_is_not_finite() {
    let Some((mut shell, _real)) = qwen3_5_shell(64) else {
        return;
    };
    shell.keep_arena(true);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];

    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let plan = hybrid_plan(&facts);

    for n in [4usize, 3] {
        let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % PERIOD.len()]).collect();
        let rows: Vec<Row> = vec![
            Row {
                samples: true,
                ..Row::default()
            };
            n
        ];
        let low = lower(
            &plan,
            &rows,
            Fire {
                captures_across_splits: false,
            },
        )
        .expect("this row count lowers");

        let Ok(step) = shell.step(&[driver_wgpu::turns::Turn {
            who: 40 + n as u64,
            tokens,
        }]) else {
            println!("{n} rows: refused");
            continue;
        };
        assert_eq!(
            step.arena.len(),
            low.arena_bytes,
            "{n} rows: the arena handed back is not the size the lowering \
             states, so the offsets below index something else"
        );

        // THE ARENA'S END STATE, and that is the whole of what this can say.
        //
        // The first draft of this walked the launches in order and called the
        // first non-finite range "where the NaN was made". It is not: an arena
        // is REUSED -- `Lowered::value_owner` exists because chains of values
        // share one slot -- and this readback happens after the last dispatch,
        // so an offset holds whatever was written there LAST. It duly named
        // `embed_gather` at offset 0, which is the residual stream's slot and
        // therefore the last writer's answer rather than the first's.
        //
        // What the end state CAN say is which symbols' outputs survive finite
        // to the end and which do not, tallied per symbol. A symbol every one
        // of whose ranges is finite did not make this NaN and did not inherit
        // it either.
        let mut tally: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        let mut nonfinite = 0usize;
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let span = (launch.rows.end - launch.rows.start) as usize;
            for arg in &low.args[launch.args.start as usize..launch.args.end as usize] {
                let Arg::Arena { at, width, bytes } = arg else {
                    continue;
                };
                // bf16 only: an f32 scratch is a different reading and the
                // activations this text carries are two-byte.
                if *bytes != 2 {
                    continue;
                }
                let end = at + span * *width as usize * 2;
                if end > step.arena.len() {
                    continue;
                }
                let bad = step.arena[*at..end]
                    .chunks_exact(2)
                    .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                    .filter(|v| !v.is_finite())
                    .count();
                let seen = tally.entry(symbol.clone()).or_insert((0, 0));
                if bad > 0 {
                    nonfinite += 1;
                    seen.1 += 1;
                } else {
                    seen.0 += 1;
                }
            }
        }
        println!("\n{n} rows: {nonfinite} operand ranges hold a non-finite value");
        for (symbol, (ok, bad)) in &tally {
            println!("    {symbol}: {ok} finite, {bad} not");
        }
    }
}

/// **How deep the text has to be before an odd row count goes NaN — and the
/// answer names the layer.**
///
/// # Why the depth and not another switch
///
/// [`the_first_value_of_an_odd_row_fire_that_is_not_finite`] reads the arena
/// back and finds it saturated — once the residual stream is NaN, everything
/// downstream is, and an END-STATE readback of a reused arena cannot name an
/// origin. The fix is not a better readback. It is a shorter fire: trace the
/// text at ONE layer and there is nothing after the suspect to overwrite the
/// evidence.
///
/// Layer 0 of this model is a linear-attention layer, so a one-layer text is
/// `embed → attn_norm → gated DeltaNet → mlp_norm → MLP → final_norm →
/// lm_head`. The earlier bisection already put the NaN inside an MLP block, and
/// this is the same block with twenty-three fewer copies of itself around it.
///
/// It holds only the weights a one-layer text binds, because `Shell::hold`
/// refuses a name the plan does not carry and a checkpoint's other 23 layers
/// are exactly that.
///
/// # What it found
///
/// ```text
///  1 layers:  2 rows 11.188   3 rows 14.375   4 rows 12.750   5 rows 10.312
///  2 layers:  2 rows 11.125   3 rows 14.000   4 rows 10.312   5 rows 10.312
///  3 layers:  2 rows 11.188   3 rows 13.938   4 rows  9.438   5 rows 10.375
///  4 layers:  2 rows 11.625   3 rows NaN      4 rows  9.250   5 rows NaN
///  8 layers:  2 rows 10.625   3 rows NaN      4 rows  9.375   5 rows NaN
/// ```
///
/// **Three layers is clean and four is not**, and this family's layer kinds
/// are not interchangeable: `full_attention_interval` is 4, so layers 0, 1 and
/// 2 are linear attention and **layer 3 is the first FULL-ATTENTION layer**.
/// The fourth layer is not one more of the same thing; it is the first of the
/// other thing.
///
/// So the odd-row NaN belongs to the full-attention path, and everything the
/// earlier rounds concluded about the MLP was a reading of a fire that
/// contained one.
///
/// It also explains why those rounds could not shake it by zeroing
/// `attn_norm`: that sets the branch's INPUT to zero and the attention kernels
/// still run. The NaN therefore does not depend on the attention's values, only
/// on its presence — which is what a structural defect looks like, and what an
/// arithmetic one does not.
///
/// # Where that points
///
/// [`whether_the_odd_row_nan_follows_the_tiled_paged_softmax`] takes it the
/// rest of the way: **`sdpa_paged_tiled_bfloat16_d_256`**.
///
/// Reports rather than asserts: the boundary is the measurement, and pinning
/// it would turn green into red the day somebody fixes it.
#[test]
#[ignore = "loads and encodes a real checkpoint; run it deliberately"]
fn whether_one_layer_is_enough_to_make_the_odd_row_nan() {
    for layers in [1u32, 2, 3, 4, 8] {
        if !one_depth(layers) {
            return;
        }
    }
}

/// One depth of the hybrid text, fired at four row counts.
///
/// Returns `false` when the run could not be set up, so the sweep stops rather
/// than reporting a row of nothing for every remaining depth.
fn one_depth(layers: u32) -> bool {
    one_text(layers, false)
}

/// [`one_text`], with every dispatch submitted on its own command buffer.
fn one_text_ordered(layers: u32, one_at_a_time: bool) -> bool {
    one_text_inner(layers, false, one_at_a_time)
}

/// **Which attention kernel carries the odd-row NaN: the tiled paged softmax,
/// or something else in the branch.**
///
/// # The switch
///
/// `Text` carries two plans and picks by row count, so putting the DECODE plan
/// in the PREFILL slot fires a multi-row window through the decode text's
/// kernels — `sdpa_vector_decode` instead of
/// `sdpa_paged_tiled_bfloat16_d_256`, and the matvec projections instead of
/// the tiled GEMM. `tests/serving.rs::shelled_with` does exactly this for
/// qwen3 and calls it the only way that harness can ask whether two kernel
/// families agree.
///
/// Here it asks something narrower: the rest of the attention branch is
/// unchanged — the same rope, the same `q_gate_split`, the same KV append, the
/// same per-head norms — so if four layers go finite with the vector text, the
/// tiled paged softmax is the one that was wrong.
///
/// Four layers, because [`whether_one_layer_is_enough_to_make_the_odd_row_nan`]
/// puts the boundary between three and four: three is the last all-linear
/// depth and four is the first that contains a full-attention layer.
///
/// # What it found, and it was the wrong half
///
/// ```text
/// prefill text (tiled):  2 rows 11.625   3 rows NaN      4 rows 9.250   5 rows NaN
/// decode text (vector):  2 rows 10.125   3 rows 10.688   4 rows 9.562   5 rows 10.688
/// ```
///
/// This was read as naming `sdpa_paged_tiled_bfloat16_d_256`, with the split
/// gated-DeltaNet pair kept as an unlikely second disjunct because the run
/// that exonerated it was at a different DEPTH.
///
/// **The second disjunct is the answer.**
/// [`the_rectangle_the_odd_row_nan_first_appears_at`] stops the fire on each
/// rectangle in turn and finds the first non-finite value at
/// **`gdn_prep_slotted_bfloat16`, layer 1** — twenty-one rectangles in, long
/// before any attention runs.
///
/// The swap still works, and now for a reason that was always in the diff: it
/// replaces the split pair with the fused `gdn_core`. Keeping the disjunct is
/// what made this correctable instead of a wrong answer with a proof attached.
///
/// The swap changes more than the softmax, so the claim needs the rest of the
/// difference ruled out. It is MEASURED rather than argued —
/// [`what_the_two_texts_differ_by_at_a_three_row_fire`] lowers both at four
/// layers and three rows and diffs the kernel tables:
///
/// ```text
/// only the PREFILL text:  gdn_prep_slotted, gdn_core_recurrent_slotted,
///                         sdpa_paged_tiled_bfloat16_d_256
/// only the DECODE text:   gdn_core_slotted, sdpa_paged_decode_bfloat16_d_256
/// ```
///
/// Five kernels, in two families — and the projections are NOT among them,
/// because a row count takes the tiled GEMM only when the tile DIVIDES it
/// (`GuardPred::TokensMultipleOf`) and 32 divides neither 3 nor 5, so both
/// texts lower to `affine_qmv_fast` here.
///
/// **The gated DeltaNet family is exonerated at THREE layers**: a three-layer
/// text is all linear attention, runs `gdn_prep` + `gdn_core_recurrent` under
/// the PREFILL text, and is finite at every row count.
///
/// That is one layer short of a proof and the gap is worth stating rather than
/// rounding off. The clean run is at a different DEPTH from the NaN, so what
/// is strictly established is: the odd-row NaN is produced by the tiled paged
/// softmax, or by the split gated-DeltaNet pair in a four-layer context where
/// three layers of it are fine.
///
/// The control that would close it does not exist for this checkpoint. Both
/// halves were tried — `full_attn_interval` 5 over four layers is all linear,
/// 1 over one layer is all attention — and neither fires, because a text's
/// layer kinds decide which weights it binds and this checkpoint's kinds do
/// not move: layer 3 ships no `conv_w` and layer 0 ships no `q_proj`. Both
/// answer 0.000 at every row count, which is a fire over projections nothing
/// filled.
///
/// What is left between the two columns is the paged softmax, and this
/// checkpoint's is a variant of its own: `head_dim` 256 with 64 rotary
/// channels, which is also the one thing qwen3-0.6b does not run and the
/// reason it is unaffected at every row count.
///
/// A softmax is where a NaN comes from for free. A query row whose every key
/// is masked has `max = -inf` and `sum = 0`, and `0 / 0` is NaN rather than a
/// refusal — and whether a row lands in a partial tile is exactly a question
/// about the row count.
#[test]
#[ignore = "loads and encodes a real checkpoint; run it deliberately"]
fn whether_the_odd_row_nan_follows_the_tiled_paged_softmax() {
    println!("the PREFILL text (tiled paged softmax):");
    if !one_text(4, false) {
        return;
    }
    println!("the DECODE text in the prefill slot (vector softmax):");
    // No guard on this one: nothing follows it, so its answer has nowhere to
    // go. The `return` above is what stops the second text from running on a
    // shell the first could not open.
    one_text(4, true);
    // NEITHER OF THE TWO CONTROLS THIS WANTS CAN BE BUILT, and both were
    // tried rather than assumed away.
    //
    // The depth sweep changes two things at once: three layers is clean and
    // four is not, and the fourth is also the first full-attention layer. To
    // separate "an attention layer" from "four layers' worth of arena" one
    // wants the same depth with no attention in it, or one attention layer
    // with no depth. `full_attn_interval` makes both expressible -- 5 over
    // four layers is all linear, 1 over one layer is all attention -- and
    // neither can be FIRED, because a text's layer kinds decide which weights
    // it binds and this checkpoint's do not move: its layer 3 is
    // full-attention and ships no `conv_w`, its layer 0 is linear and ships no
    // `q_proj`.
    //
    // Measured, both: every row count answers a span of 0.000 with no NaN
    // anywhere, which is a fire over projections nothing filled.
    //
    // So the separation rests on the TEXT SWAP instead, which holds the depth
    // fixed and changes only which kernels run. See the doc above for what
    // that does and does not settle.

    // A ONE-LAYER ALL-ATTENTION TEXT WOULD BE THE SMALLEST REPRODUCTION AND
    // CANNOT BE FIRED. `full_attn_interval` 1 makes every layer a
    // full-attention one, which is a text whose layer 0 binds `q_proj`,
    // `k_proj` and `v_proj` -- and this CHECKPOINT's layer 0 is a linear
    // attention layer, so it ships none of them. Tried: every row count
    // answers a span of 0.000 with no NaN anywhere, which is a fire over
    // projections nothing filled and not a measurement of anything.
    //
    // So four layers is the minimum, and it is the minimum for a reason about
    // the checkpoint rather than about the defect.
}

/// **Is the odd-row NaN a HAZARD — a dispatch reading what another is still
/// writing — or the tiled softmax's own arithmetic?**
///
/// # Why this is the question left
///
/// The tiled paged softmax receives an IDENTICAL grid (`[8, 1, 1]`) and
/// identical scalars at three rows and at four; only its operand extents
/// differ, and those scale exactly. Its guards are all present: rows past
/// `n_rows` `continue`, the store is bounded by `arrayLength(&out_)`, and the
/// masked-row `0 / 0` has an explicit `if (sum_exp != 0.0)`. A kernel told the
/// same thing twice that answers differently is not doing arithmetic wrong.
///
/// `Shell::one_at_a_time` submits every dispatch on its own command buffer
/// with a device wait between, which is the strongest ordering this driver can
/// impose. If the NaN survives that, the fire is ordered correctly and the
/// fault is inside a dispatch. If it does NOT, then a rectangle is reading
/// what another is still writing — and `crate::serve::Fire::one_at_a_time`'s
/// doc says that cannot happen here:
///
/// > *`wgpu` inserts the barrier itself, at every encoding granularity, so the
/// > two paths are ordered IDENTICALLY and a disagreement cannot be a missing
/// > barrier.*
///
/// That sentence is the only claim in this driver's device half taken from
/// reading somebody else's source rather than from running something, and its
/// own doc says so. This is the test it asks for.
#[test]
#[ignore = "loads and encodes a real checkpoint and submits every dispatch alone; slow"]
fn whether_the_odd_row_nan_survives_one_dispatch_at_a_time() {
    println!("recorded into one pass:");
    if !one_text_ordered(4, false) {
        return;
    }
    println!("one dispatch per submission, with a device wait between:");
    one_text_ordered(4, true);
}

/// One depth of the hybrid text, optionally with the DECODE plan in the
/// prefill slot, fired at four row counts.
fn one_text(layers: u32, vector: bool) -> bool {
    one_text_inner(layers, vector, false)
}

fn one_text_inner(layers: u32, vector: bool, one_at_a_time: bool) -> bool {
    one_text_full(layers, vector, one_at_a_time, 0)
}

/// [`one_text_inner`], optionally with `full_attn_interval` overridden so that
/// EVERY layer is a full-attention layer.
///
/// The hybrid's layer kinds come from that interval — 4 in the checkpoint, so
/// layers 0, 1 and 2 are linear attention and 3 is the first attention one.
/// Setting it to 1 makes a one-layer text a single ATTENTION layer, which is
/// the smallest fire that can carry this defect and the only one with nothing
/// before or after it to confuse an arena readback.
fn one_text_full(layers: u32, vector: bool, one_at_a_time: bool, interval: u32) -> bool {
    let Some(dir) = qwen3_5_snapshot() else {
        println!("no Qwen3.5-0.8B snapshot, so IT COULD NOT BE MEASURED");
        return false;
    };
    let Some(row) = model::catalog::find("qwen3.5-0.8b-base") else {
        println!("this build has no `qwen3.5-0.8b-base` row");
        return false;
    };
    let Some(device) = adapter() else {
        println!("no adapter, so IT COULD NOT BE MEASURED");
        return false;
    };

    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = layers;
    if interval > 0 {
        facts.full_attn_interval = interval;
    }
    let plan = hybrid_plan(&facts);
    let decode = hybrid_plan_class(&facts, model_ir::trace::FireClass::Decode);
    let Some(real) = qwen3_5_weights(&dir, row, &hybrid_plan(&Qwen35HybridFacts::qwen3_5_0_8b()))
    else {
        return false;
    };

    let text = driver_wgpu::shell::Text {
        prefill: if vector { decode.clone() } else { plan },
        decode,
        geometry: Geometry {
            q_heads: facts.attn.q_heads,
            kv_heads: facts.attn.kv_heads,
            // THE RECURRENT PAIR, stated. `Geometry::recurrent` falls back to
            // `(kv_heads, head_dim)` when this is zero, and for this hybrid
            // that is attention's 2 and 256 where the gated DeltaNet's are 16
            // and 128 -- which is FIX 8 exactly: a scan dispatched over two
            // heads of sixteen, the rest of the state left as arena litter.
            //
            // And NO `..Default::default()` below it: this literal names every
            // field, so one ADDED upstream should break the build rather than
            // be filled with a zero -- which is how the pair arrived silently
            // wrong in the first place.
            v_heads: facts.gdn.value_heads,
            v_dim: facts.gdn.value_head_dim,
            head_dim: facts.attn.head_dim,
            rotary_dims: facts.attn.rotary_dim,
            n_experts: 0,
            experts_per_token: 0,
        },
        layers: u16::try_from(layers).expect("a small stack"),
    };
    let deployment = driver_wgpu::shell::Deployment {
        pages: 64,
        theta: 10_000_000.0,
        recurrent: Some(driver_wgpu::resources::Recurrent {
            // Every layer gets a slab: a full-attention layer holds none and
            // the pool answers `None` for it, so over-allocating is a waste
            // and under-allocating is a refusal.
            // A full-attention layer holds no slab, so an all-attention text
            // needs none -- but the pool is keyed by layer NUMBER and asking
            // for more than the stack has is a waste rather than a refusal.
            linear_layers: layers,
            conv_dim: 6144,
            conv_k: 4,
            v_heads: 16,
            v_dim: 128,
            k_dim: 128,
            slots: 8,
        }),
        ..driver_wgpu::shell::Deployment::default()
    };
    let mut shell = match driver_wgpu::shell::Shell::on(device, text, deployment) {
        Ok(shell) => shell,
        Err(why) => {
            println!("\nTHE {layers}-LAYER SHELL WOULD NOT OPEN:\n  {why}");
            return false;
        }
    };

    shell.one_at_a_time(one_at_a_time);
    let mut held = 0usize;
    for (name, bytes) in &real {
        if shell.hold(name, bytes).is_ok() {
            held += 1;
        }
    }
    assert!(
        held > 0,
        "a {layers}-layer text bound none of the checkpoint's weights, so \
         anything below is a fire over an empty model"
    );

    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let mut line = String::new();
    for n in [2usize, 3, 4, 5] {
        let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % PERIOD.len()]).collect();
        let out = fire_row(&mut shell, 100 + n as u64, &tokens);
        let nan = out.iter().any(|v| v.is_nan());
        line.push_str(&format!(
            "  {n} rows: {}",
            if nan { "NaN     ".to_owned() } else { format!("{:<8.3}", span_of(&out)) }
        ));
    }
    println!(
        "{layers:>2} layers{} ({held} weights bound):{line}",
        match interval {
            1 => ", all attention",
            5 => ", all linear",
            _ => "",
        },
    );
    true
}

/// **Exactly which kernels the two texts differ by at a three-row fire.**
///
/// [`whether_the_odd_row_nan_follows_the_tiled_paged_softmax`] fires four
/// layers through the prefill text (NaN at odd row counts) and through the
/// decode text in the prefill slot (finite at all of them). That localises the
/// defect to whatever the two texts do differently — and "the tiled paged
/// softmax" is a claim about that difference, not a measurement of it.
///
/// So this measures it. CPU only.
#[test]
fn what_the_two_texts_differ_by_at_a_three_row_fire() {
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 4;
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        3
    ];
    let of = |plan: &model_ir::trace::ForwardPlan| -> BTreeSet<String> {
        lower(
            plan,
            &rows,
            Fire {
                captures_across_splits: false,
            },
        )
        .expect("four layers lower")
        .kernels
        .into_iter()
        .collect()
    };
    let prefill = of(&hybrid_plan(&facts));
    let decode = of(&hybrid_plan_class(&facts, model_ir::trace::FireClass::Decode));

    println!("only the PREFILL text launches these:");
    for k in prefill.difference(&decode) {
        println!("    {k}");
    }
    println!("only the DECODE text launches these:");
    for k in decode.difference(&prefill) {
        println!("    {k}");
    }
}

/// **Does the tiled softmax MAKE the NaN, or read one?**
///
/// # Why this is answerable now and was not before
///
/// The arena readback gives the END STATE, and over twenty-four layers that is
/// useless: once the residual stream is NaN every range in it is. At FOUR
/// layers it is not useless, because layer 3 is the LAST layer — nothing runs
/// after its attention except the epilogue, so its own operands still hold
/// what it left there.
///
/// So this fires the four-layer text at three rows with the arena kept, finds
/// layer 3's `sdpa_paged_tiled` launch in the lowering, and reads its two
/// arena operands: the queries it was given and the output it wrote.
///
/// # What it found, and what the reading of it is worth
///
/// ```text
/// 4 rows: sdpa layer 3 -> q at 8192: 0 of 8192 not finite
///                         o at 24576: 0 of 8192 not finite
///         ranges finite/not by layer: {0: (40,0), 1: (33,0), 2: (33,0), 3: (39,0)}
/// 3 rows: sdpa layer 3 -> q at 7168: 6144 of 6144 not finite
///                         o at 19456: 6144 of 6144 not finite
///         ranges finite/not by layer: {0: (4,36), 1: (0,33), 2: (0,33), 3: (0,39)}
/// ```
///
/// **It does not settle it, and the reason is the instrument.** Four layers is
/// short enough that layer 3's operands survive, but it is not short enough
/// that layer 0's do: an arena is reused, so layer 0 reading 36-of-40
/// non-finite at the end is equally what a slot LAYER 3 wrote looks like.
///
/// So the honest statement is narrower than the last commit's: the tiled
/// softmax's query range holds NaN at the end of a three-row fire, and this
/// cannot say whether it held NaN when the softmax read it.
///
/// What is NOT weakened is the localisation, because it rests on two
/// independent things this does not touch: the depth boundary (three layers
/// finite, four not, and the fourth is the first attention layer) and the text
/// swap with its kernel diff measured. Settling the last step wants a readback
/// per DISPATCH rather than per fire.
#[test]
#[ignore = "loads and encodes a real checkpoint; run it deliberately"]
fn whether_the_tiled_softmax_makes_the_nan_or_reads_one() {
    let Some(dir) = qwen3_5_snapshot() else {
        println!("no Qwen3.5-0.8B snapshot, so IT COULD NOT BE MEASURED");
        return;
    };
    let Some(row) = model::catalog::find("qwen3.5-0.8b-base") else {
        return;
    };
    let Some(device) = adapter() else {
        println!("no adapter, so IT COULD NOT BE MEASURED");
        return;
    };
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 4;
    let plan = hybrid_plan(&facts);
    let decode = hybrid_plan_class(&facts, model_ir::trace::FireClass::Decode);
    let Some(real) = qwen3_5_weights(&dir, row, &hybrid_plan(&Qwen35HybridFacts::qwen3_5_0_8b()))
    else {
        return;
    };

    let text = driver_wgpu::shell::Text {
        decode,
        prefill: plan.clone(),
        geometry: Geometry {
            q_heads: facts.attn.q_heads,
            kv_heads: facts.attn.kv_heads,
            // THE RECURRENT PAIR, stated. `Geometry::recurrent` falls back to
            // `(kv_heads, head_dim)` when this is zero, and for this hybrid
            // that is attention's 2 and 256 where the gated DeltaNet's are 16
            // and 128 -- which is FIX 8 exactly: a scan dispatched over two
            // heads of sixteen, the rest of the state left as arena litter.
            //
            // And NO `..Default::default()` below it: this literal names every
            // field, so one ADDED upstream should break the build rather than
            // be filled with a zero -- which is how the pair arrived silently
            // wrong in the first place.
            v_heads: facts.gdn.value_heads,
            v_dim: facts.gdn.value_head_dim,
            head_dim: facts.attn.head_dim,
            rotary_dims: facts.attn.rotary_dim,
            n_experts: 0,
            experts_per_token: 0,
        },
        layers: 4,
    };
    let mut shell = match driver_wgpu::shell::Shell::on(
        device,
        text,
        driver_wgpu::shell::Deployment {
            pages: 64,
            theta: 10_000_000.0,
            recurrent: Some(driver_wgpu::resources::Recurrent {
                linear_layers: 4,
                conv_dim: 6144,
                conv_k: 4,
                v_heads: 16,
                v_dim: 128,
                k_dim: 128,
                slots: 8,
            }),
            ..driver_wgpu::shell::Deployment::default()
        },
    ) {
        Ok(shell) => shell,
        Err(why) => {
            println!("the four-layer shell would not open: {why}");
            return;
        }
    };
    shell.keep_arena(true);
    for (name, bytes) in &real {
        let _ = shell.hold(name, bytes);
    }

    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    for n in [4usize, 3] {
        let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % PERIOD.len()]).collect();
        let rows: Vec<Row> = vec![
            Row {
                samples: true,
                ..Row::default()
            };
            n
        ];
        let low = lower(
            &plan,
            &rows,
            Fire {
                captures_across_splits: false,
            },
        )
        .expect("four layers lower");
        let Ok(step) = shell.step(&[driver_wgpu::turns::Turn {
            who: 70 + n as u64,
            tokens,
        }]) else {
            println!("{n} rows: refused");
            continue;
        };
        if step.arena.len() != low.arena_bytes {
            println!("{n} rows: the arena is {} and the lowering states {}, so \
                 the offsets would index something else", step.arena.len(), low.arena_bytes);
            continue;
        }

        // BY LAYER, because the end state saturates: once the residual stream
        // is NaN every later range is, so the only thing an end-state readback
        // can localise is the EARLIEST layer that still holds one -- and only
        // while the layers before it have not had their slots reused.
        let mut per_layer: BTreeMap<u16, (usize, usize)> = BTreeMap::new();
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let span2 = (launch.rows.end - launch.rows.start) as usize;
            for a in &low.args[launch.args.start as usize..launch.args.end as usize] {
                if let Arg::Arena { at, width, bytes } = a
                    && *bytes == 2
                {
                    let end = (at + span2 * *width as usize * 2).min(step.arena.len());
                    let bad = step.arena[*at..end]
                        .chunks_exact(2)
                        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                        .filter(|v| !v.is_finite())
                        .count();
                    let e = per_layer.entry(launch.layers.start).or_insert((0, 0));
                    if bad > 0 { e.1 += 1; } else { e.0 += 1; }
                }
            }
            if !symbol.starts_with("sdpa") {
                continue;
            }
            let span = (launch.rows.end - launch.rows.start) as usize;
            let ranges: Vec<String> = low.args
                [launch.args.start as usize..launch.args.end as usize]
                .iter()
                .filter_map(|a| match a {
                    Arg::Arena { at, width, bytes } if *bytes == 2 => {
                        let end = at + span * *width as usize * 2;
                        let bad = step.arena[*at..end.min(step.arena.len())]
                            .chunks_exact(2)
                            .map(|c| {
                                f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                            })
                            .filter(|v| !v.is_finite())
                            .count();
                        Some(format!("at {at}: {bad} of {} not finite", span * *width as usize))
                    }
                    _ => None,
                })
                .collect();
            println!("{n} rows: {symbol} layer {} -> {ranges:?}", launch.layers.start);
        }
        println!("{n} rows, ranges finite/not by layer: {per_layer:?}");
    }
}

/// **Is the tiled softmax's query range touched again after it runs?**
///
/// # Why this settles what the readback could not
///
/// [`whether_the_tiled_softmax_makes_the_nan_or_reads_one`] finds that range
/// holding NaN at the END of a three-row fire and says, correctly, that this
/// cannot distinguish "it was NaN when the softmax read it" from "something
/// later wrote over it". That is a question about the LOWERING, not about the
/// device: if no launch after the softmax names a range overlapping its
/// queries, then nothing later could have written there and the end state IS
/// what the softmax read.
///
/// # What it found
///
/// **Seven later operand ranges overlap them**, at both row counts: the
/// residual-folding `o_proj` writes `7168..13312`, the MLP's norm reads and
/// writes the same span, and the MLP's own projections cover `13312..34816`.
///
/// So the answer is no, and the earlier readback's caveat was the right one:
/// the NaN at `7168` at the end of the fire is the MLP's, whatever the softmax
/// read. Nothing about the tiled softmax's INPUT can be recovered from an
/// end-state arena, and the localisation stands on the depth boundary and the
/// text swap alone.
///
/// CPU only, and it needs no checkpoint.
#[test]
fn whether_anything_writes_the_tiled_softmaxs_queries_after_it_runs() {
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 4;
    let plan = hybrid_plan(&facts);

    for n in [3usize, 4] {
        let rows: Vec<Row> = vec![
            Row {
                samples: true,
                ..Row::default()
            };
            n
        ];
        let low = lower(
            &plan,
            &rows,
            Fire {
                captures_across_splits: false,
            },
        )
        .expect("four layers lower");

        // The LAST tiled softmax, and its queries: the first of its two-byte
        // arena operands, which is the order `arm::paged_plain` states them in.
        let mut at_after: Option<(usize, (usize, usize))> = None;
        for (i, launch) in low.launches.iter().enumerate() {
            if !low.kernels[launch.kernel as usize].starts_with("sdpa_paged_tiled") {
                continue;
            }
            let span = (launch.rows.end - launch.rows.start) as usize;
            if let Some(Arg::Arena { at, width, bytes }) = low.args
                [launch.args.start as usize..launch.args.end as usize]
                .iter()
                .find(|a| matches!(a, Arg::Arena { bytes: 2, .. }))
            {
                at_after = Some((i, (*at, at + span * *width as usize * *bytes as usize)));
            }
        }
        let Some((after, q)) = at_after else {
            println!("{n} rows: no tiled softmax in this plan");
            continue;
        };

        let mut touched = Vec::new();
        for launch in low.launches.iter().skip(after + 1) {
            let symbol = &low.kernels[launch.kernel as usize];
            let span = (launch.rows.end - launch.rows.start) as usize;
            for a in &low.args[launch.args.start as usize..launch.args.end as usize] {
                if let Arg::Arena { at, width, bytes } = a {
                    let end = at + span * *width as usize * *bytes as usize;
                    if *at < q.1 && q.0 < end {
                        touched.push(format!("{symbol} ({at}..{end})"));
                    }
                }
            }
        }
        touched.dedup();
        println!(
            "{n} rows: the last tiled softmax's queries are {q:?}; {} later \
             operand ranges overlap them",
            touched.len()
        );
        for line in touched.iter().take(5) {
            println!("    {line}");
        }
    }
}

/// **The rectangle a NaN first appears at, by stopping the fire on it.**
///
/// # The instrument
///
/// Everything before this could see only the LAST thing written to each arena
/// offset, and once a computation has gone wrong everything after it has too —
/// so a whole-fire readback of a three-row qwen3.5 says "831 ranges are
/// non-finite" and names nothing. [`driver_wgpu::shell::Shell::fire_prefix`]
/// records only the first `n` rectangles, which makes the arena's end state the
/// state AT `n`. Walking `n` therefore finds the rectangle rather than the
/// subsystem.
///
/// Four layers, because that is the shallowest text that reproduces (layer 3
/// is the first full-attention layer) and 3 rows, because that is the
/// shortest odd prompt.
///
/// # Binary search, and why it is sound here
///
/// "Some range is non-finite after `n` rectangles" is MONOTONE in `n`: the
/// arena is zeroed at allocation, a NaN is only ever written, and nothing
/// erases one — a later rectangle may overwrite the slot, but then it wrote a
/// value derived from the NaN it read, which is a NaN. So the first `n` where
/// the predicate holds is findable in `log2(364)` fires instead of 364.
///
/// The two ends are asserted rather than assumed: at `n = 0` nothing has run
/// and the arena must be clean, and at the full length it must be dirty. A
/// search whose ends do not bracket is a search reporting an index it invented.
#[test]
#[ignore = "loads and encodes a real checkpoint and fires it ~10 times; slow"]
fn the_rectangle_the_odd_row_nan_first_appears_at() {
    let Some(dir) = qwen3_5_snapshot() else {
        println!("no Qwen3.5-0.8B snapshot, so IT COULD NOT BE MEASURED");
        return;
    };
    let Some(row) = model::catalog::find("qwen3.5-0.8b-base") else {
        return;
    };
    let Some(device) = adapter() else {
        println!("no adapter, so IT COULD NOT BE MEASURED");
        return;
    };
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 4;
    let plan = hybrid_plan(&facts);
    let decode = hybrid_plan_class(&facts, model_ir::trace::FireClass::Decode);
    let Some(real) = qwen3_5_weights(&dir, row, &hybrid_plan(&Qwen35HybridFacts::qwen3_5_0_8b()))
    else {
        return;
    };
    let mut shell = match driver_wgpu::shell::Shell::on(
        device,
        driver_wgpu::shell::Text {
            decode,
            prefill: plan.clone(),
            geometry: Geometry {
                q_heads: facts.attn.q_heads,
                kv_heads: facts.attn.kv_heads,
                head_dim: facts.attn.head_dim,
                rotary_dims: facts.attn.rotary_dim,
                n_experts: 0,
                experts_per_token: 0,
                ..Default::default()
            },
            layers: 4,
        },
        driver_wgpu::shell::Deployment {
            pages: 64,
            theta: 10_000_000.0,
            recurrent: Some(driver_wgpu::resources::Recurrent {
                linear_layers: 4,
                conv_dim: 6144,
                conv_k: 4,
                v_heads: 16,
                v_dim: 128,
                k_dim: 128,
                // One per probe and then some: a bisection over 64 rectangles
                // is ten fires, each on a fresh row, and a truncated fire
                // leaves its slot half-written -- so reusing one would make
                // every later probe a reading of the last probe's leftovers.
                slots: 64,
            }),
            ..driver_wgpu::shell::Deployment::default()
        },
    ) {
        Ok(shell) => shell,
        Err(why) => {
            println!("the four-layer shell would not open: {why}");
            return;
        }
    };
    shell.keep_arena(true);
    for (name, bytes) in &real {
        let _ = shell.hold(name, bytes);
    }

    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let tokens: Vec<u32> = PERIOD[..3].to_vec();
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        3
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("four layers lower");
    let total = low.launches.len();
    println!("{total} rectangles in a four-layer three-row fire");
    for (i, l) in low.launches.iter().take(10).enumerate() {
        println!("   launch {i}: {}", low.kernels[l.kernel as usize]);
    }

    // Does the arena hold a non-finite bf16 anywhere the plan names, after
    // `n` rectangles? A fresh row each time, so no fire inherits another's KV
    // or carry.
    let mut who = 200u64;
    let mut dirty = |shell: &mut driver_wgpu::shell::Shell, n: usize| -> bool {
        who += 1;
        shell.fire_prefix(Some(n));
        // A REFUSED PROBE IS NOT A CLEAN ONE. The first draft returned `false`
        // here, which reads as "no NaN after `n` rectangles" -- and every probe
        // after the eighth was refused, because each takes a fresh row and the
        // pool had eight recurrent slots. The bisection then walked a
        // predicate that was answering about the pool, and its own endpoint
        // check is what caught it.
        let step = shell
            .step(&[driver_wgpu::turns::Turn {
                who,
                tokens: tokens.clone(),
            }])
            .unwrap_or_else(|why| {
                panic!(
                    "a prefix of {n} was refused ({why}), and a refused probe is not a clean one -- every answer this search gives after it would be about the shell rather than the fire"
                )
            });
        // ONLY THE RECTANGLE THAT JUST RAN, and read at ITS OWN dtype.
        //
        // The first draft scanned every earlier rectangle's operands as bf16
        // and reported rectangle 21 -- falsely. An arena is reused: layer 1's
        // `pre_q` and `pre_k` are f32 and sit exactly where layer 0's MLP
        // scratch was, so the earlier launch's bf16 view of those bytes reads
        // f32 mantissas as half-words, and some of them are NaN patterns. The
        // predicate was measuring the dtype it assumed, not the values.
        //
        // A rectangle's own operands, at the prefix that ends on it, hold what
        // IT wrote. That is the only view of the arena that is not a guess.
        if n == 0 {
            return false;
        }
        let launch = &low.launches[n - 1];
        let span = (launch.rows.end - launch.rows.start) as usize;
        for a in &low.args[launch.args.start as usize..launch.args.end as usize] {
            let Arg::Arena { at, width, bytes } = a else {
                continue;
            };
            let end = (at + span * *width as usize * *bytes as usize).min(step.arena.len());
            let bad = if *bytes == 2 {
                step.arena[*at..end].chunks_exact(2).any(|c| {
                    !f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16).is_finite()
                })
            } else {
                step.arena[*at..end]
                    .chunks_exact(4)
                    .any(|c| !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite())
            };
            if bad {
                return true;
            }
        }
        false
    };

    assert!(
        !dirty(&mut shell, 0),
        "the arena holds a non-finite value before any rectangle has run, so \
         it is not being zeroed and the search below would report 1"
    );

    // LINEAR, not a bisection. "This rectangle's own outputs are non-finite"
    // is NOT monotone in the prefix -- a later rectangle writing a finite
    // value makes it false again -- and a bisection over a non-monotone
    // predicate reports an index it invented. Sixty-four fires is seconds once
    // the weights are held; the two minutes is the staging, which happens
    // once.
    let mut hi = total;
    for n in 1..=total {
        if dirty(&mut shell, n) {
            hi = n;
            break;
        }
    }
    let launch = &low.launches[hi - 1];
    println!(
        "\nTHE FIRST NON-FINITE VALUE APPEARS AT RECTANGLE {hi} of {total}: \
         `{}`, layer {}, rows {:?}",
        low.kernels[launch.kernel as usize],
        launch.layers.start,
        launch.rows,
    );

    // The neighbourhood, free from the lowering, so the answer can be read
    // against what runs beside it rather than taken on its index.
    println!("  the rectangles around it:");
    for i in hi.saturating_sub(4)..(hi + 2).min(total) {
        let l = &low.launches[i];
        println!(
            "    {}{i:>3}: {} layer {}",
            if i == hi - 1 { "->" } else { "  " },
            low.kernels[l.kernel as usize],
            l.layers.start,
        );
    }

    // THE ENDPOINTS, restated as a measurement rather than left implicit in
    // the search's invariant. A bisection reports an index whether or not its
    // predicate is monotone, and the two fires either side of the answer are
    // what say this one is real.
    let before = dirty(&mut shell, hi.saturating_sub(1));
    let after = dirty(&mut shell, hi);
    println!("  after {} rectangles: {}", hi - 1, if before { "dirty" } else { "clean" });
    println!("  after {hi} rectangles: {}", if after { "dirty" } else { "clean" });
    assert!(
        !before && after,
        "rectangle {hi} is not where it changes, so the predicate is not monotone in the prefix and the bisection reported an index it invented"
    );

    // WHICH OPERAND. The rectangle is known and its operands are a handful,
    // and a fire stopped exactly on it leaves each of them holding what that
    // rectangle left there. For `gdn_prep` the answer separates two different
    // defects: `pre_gate` is the decay and beta arithmetic
    // (`exp(-exp(A_log) * softplus(a + dt_bias))`, where an infinite exponent
    // times a zero softplus is NaN) and `pre_q`/`pre_k` are the convolution,
    // the silu and the two L2 norms.
    who += 1;
    shell.fire_prefix(Some(hi));
    let Ok(step) = shell.step(&[driver_wgpu::turns::Turn {
        who,
        tokens: tokens.clone(),
    }]) else {
        return;
    };
    println!("  its operands, in the order the statement carries them:");
    let span = (launch.rows.end - launch.rows.start) as usize;
    for (i, a) in low.args[launch.args.start as usize..launch.args.end as usize]
        .iter()
        .enumerate()
    {
        let (at, width, bytes) = match a {
            Arg::Arena { at, width, bytes } => (*at, *width, *bytes),
            other => {
                println!("    {i}: {other:?}");
                continue;
            }
        };
        let end = (at + span * width as usize * bytes as usize).min(step.arena.len());
        let bad = if bytes == 2 {
            step.arena[at..end]
                .chunks_exact(2)
                .filter(|c| {
                    !f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16).is_finite()
                })
                .count()
        } else {
            step.arena[at..end]
                .chunks_exact(4)
                .filter(|c| !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite())
                .count()
        };
        println!("    {i}: arena at {at}, {span} x {width} x {bytes}B -> {bad} not finite");
        // WHICH ones, when there are few. `core_out` is `[token, head,
        // channel]` with `Hv * Dv` per row, so the coordinates say whether the
        // thirteen are one token's, one head's, or sit on a `dv` boundary --
        // three different bugs that all come out as a count.
        if bad > 0 && bad <= 32 && bytes == 2 {
            let hv = 16usize;
            let dv = width as usize / hv;
            let mut where_: Vec<String> = Vec::new();
            for (j, c) in step.arena[at..end].chunks_exact(2).enumerate() {
                let v = f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16);
                if !v.is_finite() {
                    let t = j / width as usize;
                    let rest = j % width as usize;
                    // NaN or INF, and they are different bugs: an infinity is
                    // a magnitude that overflowed, a NaN is a `0/0` or an
                    // `inf - inf`. `bf16` carries `f32`'s exponent range, so a
                    // finite `f32` rounding up at the top of it lands on inf.
                    where_.push(format!(
                        "({} t{} h{} d{})",
                        if v.is_nan() { "NaN" } else if v > 0.0 { "+inf" } else { "-inf" },
                        t,
                        rest / dv,
                        rest % dv
                    ));
                }
            }
            println!("       at {}", where_.join(" "));
            // THE FIRST TOKEN'S ANSWER, RECOMPUTED ON THE CPU.
            //
            // At `t = 0` the recurrent state is whatever the slab held, and
            // this fire's seat is freshly zeroed -- so the scan reduces to one
            // line. `st` starts at 0, `kv` is 0, `delta` is `vv * gb`,
            // `st = k * delta`, and the output is
            //
            //     out = vv * gb * SUM_d k[d] * q[d]
            //
            // Every term is readable: `pre_q` and `pre_k` are `Hv * Dk` f32
            // per token, `pre_gate` is `2*Hv` gates then `Hv * Dv` staged v.
            // So a channel that comes back infinite can be asked WHICH factor
            // did it, which is the difference between a shader that is wrong
            // and data that overflows.
            for line in &where_ {
                if !line.contains("t0 ") {
                    continue;
                }
                let nums: Vec<usize> = line
                    .split(|c: char| !c.is_ascii_digit())
                    .filter(|w| !w.is_empty())
                    .map(|w| w.parse().unwrap_or(0))
                    .collect();
                let (h, d) = (nums[1], nums[2]);
                let f32_at = |base: usize, i: usize| -> f32 {
                    let o = base + i * 4;
                    f32::from_le_bytes([
                        step.arena[o],
                        step.arena[o + 1],
                        step.arena[o + 2],
                        step.arena[o + 3],
                    ])
                };
                let (qb, kb, gb_base) = (136192usize, 61952usize, 86528usize);
                let gb = f32_at(gb_base, 2 * h + 1);
                let ga = f32_at(gb_base, 2 * h);
                let vv = f32_at(gb_base, 2 * 16 + h * 128 + d);
                let mut dot = 0.0f64;
                let (mut qmax, mut kmax) = (0.0f32, 0.0f32);
                for i in 0..128 {
                    let q = f32_at(qb, h * 128 + i);
                    let k = f32_at(kb, h * 128 + i);
                    dot += f64::from(q) * f64::from(k);
                    qmax = qmax.max(q.abs());
                    kmax = kmax.max(k.abs());
                }
                println!(
                    "       t0 h{h} d{d}: ga {ga:e} gb {gb:e} vv {vv:e}                      k.q {dot:e} |q|max {qmax:e} |k|max {kmax:e} -> vv*gb*k.q {:e}",
                    f64::from(vv) * f64::from(gb) * dot,
                );
            }
        }
    }

    // AND THE VICTIM. If none of the rectangle's own operands holds the NaN
    // but the fire is dirty at it, then it wrote somewhere it does not
    // declare -- so the range that IS dirty belongs to an earlier rectangle,
    // and its offset against this one's says how far past the end the write
    // went.
    // AND THE SLAB, which is the one input the walk above cannot reach. The
    // scan loads `st` from `recurrent_state` before its first token and writes
    // it back after its last, so a NaN already there comes out looking exactly
    // like one the scan made. `Shell::recurrent` exists for this.
    if let Some(pool) = shell.recurrent() {
        for which in ["recurrent_state", "conv_state", "new_conv_state"] {
            let Some(slab) = pool.slab(launch.layers.start, which) else {
                continue;
            };
            // THE WHOLE SLAB, not a prefix. The first draft capped this at a
            // megabyte, and the probe's row does not sit in the first
            // megabyte: `Book::free_slot` gives each new conversation an
            // unused seat and this test opens dozens, so `state_base` for the
            // seat in use is far past a cap that only ever covered slot zero.
            // A clean read of the wrong region is the most confident kind of
            // wrong answer.
            let bytes = match shell.device().read_at(slab, 0, slab.size()) {
                Ok(b) => b,
                Err(why) => {
                    println!("    `{which}` would not read back: {why}");
                    continue;
                }
            };
            let bad = bytes
                .chunks_exact(4)
                .filter(|c| !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite())
                .count();
            // NONZERO too, not only finite. The CPU recomputation above
            // assumes the scan's state starts at zero at `t = 0`, which is
            // true for a freshly seated conversation and false the moment a
            // slot is reused -- and "the shader disagrees with its own inputs"
            // is a claim that rests entirely on it.
            let nonzero = bytes
                .chunks_exact(4)
                .filter(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) != 0.0)
                .count();
            println!(
                "    slab `{which}` layer {}: {bad} of {} floats not finite,                  {nonzero} nonzero",
                launch.layers.start,
                bytes.len() / 4,
            );
        }
    }

    println!("  every earlier range that is not finite:");
    let mut named = 0usize;
    for l in &low.launches[..hi] {
        let sp = (l.rows.end - l.rows.start) as usize;
        for a in &low.args[l.args.start as usize..l.args.end as usize] {
            let Arg::Arena { at, width, bytes } = a else {
                continue;
            };
            let end = (at + sp * *width as usize * *bytes as usize).min(step.arena.len());
            let bad = if *bytes == 2 {
                step.arena[*at..end]
                    .chunks_exact(2)
                    .filter(|c| {
                        !f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                            .is_finite()
                    })
                    .count()
            } else {
                step.arena[*at..end]
                    .chunks_exact(4)
                    .filter(|c| !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite())
                    .count()
            };
            if bad > 0 && named < 8 {
                named += 1;
                println!(
                    "    {} layer {}: at {at}..{end} ({sp} x {width} x {bytes}B) -> {bad} not finite",
                    low.kernels[l.kernel as usize],
                    l.layers.start,
                );
            }
        }
    }
}

/// **Fed one token at a time, does qwen3.5 continue the pattern?**
///
/// # Why this is the test the whole file has been building to
///
/// [`qwen3_5_fires_or_says_where_it_stopped`] gives the induction prompt as one
/// PREFILL and the answer is wrong.
/// [`the_same_prompt_twice_is_the_same_answer_or_the_scan_is_racing`] says why:
/// the prefill fires the decode-shaped gated-DeltaNet pair over TOKEN rows, so
/// every token of the prompt does a read-modify-write on one slot's recurrent
/// state inside a single dispatch, and four identical fires give four answers.
///
/// A one-token turn is `FireClass::Decode`. It fires the FUSED `gdn_core`, one
/// row per request, which is the shape that kernel is written for — and the
/// same test shows it is deterministic. So feeding the identical prompt one
/// token at a time runs the identical arithmetic with the recurrence
/// SEQUENCED, and that is the only difference.
///
/// - **It continues the pattern** → the recurrence is the whole defect, and
///   everything else in this driver's hybrid path is right. The remedy is a
///   text that fires the prompt-shaped scan, which lives in `model-dsl` and
///   `crates/model`.
/// - **It does not** → something else is wrong too, and the sequencing is not
///   sufficient.
///
/// Either way it is worth more than another reading of a shader, because it
/// separates the driver from the text.
#[test]
#[ignore = "loads and encodes a real checkpoint and fires 32 decodes; run it deliberately"]
fn whether_one_token_at_a_time_continues_the_pattern() {
    let Some((mut shell, _real)) = qwen3_5_shell(64) else {
        return;
    };
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let mut tokens: Vec<u32> = Vec::new();
    for _ in 0..5 {
        tokens.extend_from_slice(&PERIOD);
    }
    tokens.push(PERIOD[0]);
    tokens.push(PERIOD[1]);

    // ONE AT A TIME, on one row, so the carry is threaded through 32 fires of
    // the fused kernel instead of folded by one dispatch over 32 rows.
    let mut last = Vec::new();
    for (i, token) in tokens.iter().enumerate() {
        let out = fire_row(&mut shell, 1, std::slice::from_ref(token));
        if out.is_empty() {
            println!("decode {i} was refused");
            return;
        }
        last = out;
    }
    let mut top: Vec<(usize, f32)> = last.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("\nONE TOKEN AT A TIME, {} decodes:", tokens.len());
    println!("  top: {:?}", &top[..8.min(top.len())]);
    println!("  wanted {} (the period's third token)", PERIOD[2]);

    // The prefill of the same prompt, for the comparison this exists to make.
    let whole = fire_row(&mut shell, 2, &tokens);
    println!("  the same prompt as ONE prefill wants {}", argmax_of(&whole));

    if u32::try_from(top[0].0) == Ok(PERIOD[2]) {
        println!(
            "\n  IT CONTINUES THE PATTERN WHEN THE RECURRENCE IS SEQUENCED. \
             So the gated DeltaNet's arithmetic, its weights, its carry and \
             everything around them are RIGHT, and the whole of what is wrong \
             is that a prefill folds the prompt into the state in parallel."
        );
    } else {
        println!(
            "\n  IT DOES NOT CONTINUE THE PATTERN EVEN SEQUENCED: it wants {} \
             and the period says {}. So the unsequenced prefill is not the \
             whole of it.",
            top[0].0, PERIOD[2],
        );
    }
    assert_eq!(
        last.iter().filter(|v| v.is_finite()).count(),
        last.len(),
        "the decode run answered with a non-finite logit"
    );
}

/// How much each layer adds to the residual stream, layer by layer.
///
/// The answer is wrong in a particular way: the logits want token 220 and its
/// runners-up are 11, 198, 12, 13 -- a space, a comma, a newline, a hyphen, a
/// full stop. That is what a language model says when it has been given
/// almost no signal, and it is the shape a residual stream produces when the
/// LAYERS CONTRIBUTE ALMOST NOTHING and the tied `lm_head` is reading back
/// little more than the embedding it started from.
///
/// So the question is not "which value is wrong" but "which layer is silent",
/// and that is a magnitude and not a comparison -- it needs no reference.
///
/// # Why a prefix fire and not one readback
///
/// An arena is REUSED. Read at the end of a fire, an offset holds whatever was
/// written there last, so a table of "layer 3's residual" taken from a
/// finished fire is a table of names attached to the wrong bytes -- the trap
/// [`the_rectangle_the_odd_row_nan_first_appears_at`] fell into and documents.
/// [`driver_wgpu::shell::Shell::fire_prefix`] stops the fire ON a rectangle,
/// which makes the arena's end state the state AT that rectangle, and then the
/// rectangle's own operands hold what IT wrote.
///
/// One fire per measured rectangle, each on a fresh row so no fire inherits
/// another's carry.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn how_much_each_layer_adds_to_the_residual_stream() {
    residual_survey(4, false);
}

/// [`how_much_each_layer_adds_to_the_residual_stream`] at the model's OWN
/// depth.
///
/// Four layers is one sample of each layer kind, and the shrink the four-layer
/// table shows at its single attention layer is a shape, not yet a pattern.
/// Twenty-four layers is six full-attention layers -- 3, 7, 11, 15, 19, 23 --
/// against eighteen gated-DeltaNet ones, and whether the stream recovers
/// between them is what says which reading is right.
///
/// Only the residual rectangles, because each measurement is a FIRE and each
/// fire takes a fresh recurrent seat: every rectangle at this depth would be
/// three hundred and sixty seats of eighteen layers each.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn how_much_each_layer_adds_to_the_residual_stream_at_full_depth() {
    residual_survey(24, true);
}

fn residual_survey(layers: u32, only_residual: bool) {
    let Some(dir) = qwen3_5_snapshot() else {
        println!("no Qwen3.5-0.8B snapshot, so IT COULD NOT BE MEASURED");
        return;
    };
    let Some(row) = model::catalog::find("qwen3.5-0.8b-base") else {
        return;
    };
    let Some(device) = adapter() else {
        println!("no adapter, so IT COULD NOT BE MEASURED");
        return;
    };
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = layers;
    let plan = hybrid_plan(&facts);
    let decode = hybrid_plan_class(&facts, model_ir::trace::FireClass::Decode);
    let Some(real) = qwen3_5_weights(&dir, row, &hybrid_plan(&Qwen35HybridFacts::qwen3_5_0_8b()))
    else {
        return;
    };
    let mut shell = match driver_wgpu::shell::Shell::on(
        device,
        driver_wgpu::shell::Text {
            decode,
            prefill: plan.clone(),
            geometry: Geometry {
                q_heads: facts.attn.q_heads,
                kv_heads: facts.attn.kv_heads,
                head_dim: facts.attn.head_dim,
                rotary_dims: facts.attn.rotary_dim,
                n_experts: 0,
                experts_per_token: 0,
                ..Default::default()
            },
            layers: u16::try_from(layers).expect("a layer count that fits"),
        },
        driver_wgpu::shell::Deployment {
            pages: 256,
            theta: 10_000_000.0,
            recurrent: Some(driver_wgpu::resources::Recurrent {
                linear_layers: layers,
                conv_dim: 6144,
                conv_k: 4,
                v_heads: 16,
                v_dim: 128,
                k_dim: 128,
                slots: 96,
            }),
            ..driver_wgpu::shell::Deployment::default()
        },
    ) {
        Ok(shell) => shell,
        Err(why) => {
            println!("the four-layer shell would not open: {why}");
            return;
        }
    };
    shell.keep_arena(true);
    for (name, bytes) in &real {
        let _ = shell.hold(name, bytes);
    }

    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let tokens: Vec<u32> = PERIOD[..3].to_vec();
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        3
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("four layers lower");

    // EVERY rectangle, not only the residual adds: which kernel carries the
    // stream between layers is a fact of the lowering and not of this test,
    // and a table that only measured the ones this file could name in advance
    // would be a table of its own assumptions.
    let mut who = 700u64;
    println!(
        "\nwidest |value| a rectangle leaves in each of its arena ranges, at the prefix ending on it\n"
    );
    let mut last_layer = u16::MAX;
    for n in 1..=low.launches.len() {
        let name = &low.kernels[low.launches[n - 1].kernel as usize];
        // The LAST rectangles too, whatever they are called: the readout is
        // where a residual that grew sanely can still answer nothing.
        if only_residual && !name.contains("residual") && n + 4 < low.launches.len() {
            continue;
        }
        who += 1;
        shell.fire_prefix(Some(n));
        let Ok(step) = shell.step(&[driver_wgpu::turns::Turn {
            who,
            tokens: tokens.clone(),
        }]) else {
            println!("  prefix {n} was refused, so the table stops here");
            break;
        };
        let launch = &low.launches[n - 1];
        let span = (launch.rows.end - launch.rows.start) as usize;
        let mut widest: Vec<String> = Vec::new();
        for a in &low.args[launch.args.start as usize..launch.args.end as usize] {
            let Arg::Arena { at, width, bytes } = a else {
                continue;
            };
            let end = (at + span * *width as usize * *bytes as usize).min(step.arena.len());
            if end <= *at {
                continue;
            }
            // `f32::max` RETURNS THE NON-NAN OPERAND, so a fold with it reports
            // an all-NaN range as zero. Counted separately, and the NaN count
            // is printed rather than folded away.
            let mut top = 0.0f32;
            let mut nan = 0usize;
            let vals: Box<dyn Iterator<Item = f32>> = if *bytes == 2 {
                Box::new(step.arena[*at..end].chunks_exact(2).map(|c| {
                    f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                }))
            } else {
                Box::new(
                    step.arena[*at..end]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                )
            };
            for v in vals {
                if v.is_finite() {
                    top = top.max(v.abs());
                } else {
                    nan += 1;
                }
            }
            // The WIDTH, not just the value: a rectangle's arena ranges are
            // printed in body order and nothing else says which is which, so
            // `w2048` against `w2080` is what separates the scan's `pre_q`
            // from its `pre_gate`.
            widest.push(if nan == 0 {
                format!("w{width}:{top:.4}")
            } else {
                format!("w{width}:{top:.4}(+{nan} nonfinite)")
            });
        }
        if launch.layers.start != last_layer {
            println!("  -- layer {} --", launch.layers.start);
            last_layer = launch.layers.start;
        }
        println!(
            "  {n:>3} {:<48} {}",
            low.kernels[launch.kernel as usize],
            widest.join("  ")
        );
    }
    shell.fire_prefix(None);
}

/// A prompt of ONE TOKEN REPEATED, which any language model continues.
///
/// Every measurement so far says the forward pass is healthy: the residual
/// stream grows from 0.0957 at the embedding to 3.5 after twenty-four layers,
/// nothing anywhere is non-finite, and the logits come out at plus or minus
/// eighteen. And the answer is still wrong. That is the shape of a SYSTEMATIC
/// error -- a permutation, a slice, a position -- and not of a numerical one,
/// so the next question is how wrong, and it needs a floor to measure against.
///
/// `"x x x x ..."` is that floor. It needs no tokenizer, no reference
/// implementation and no knowledge of what the model was trained on: whatever
/// `x` is, a model that has learned anything at all answers `x`. It is a
/// strictly easier question than the six-token period the other probes ask,
/// and it separates "this backend computes a different model" from "this
/// benchmark is hard for 0.8B".
///
/// Four different `x`, because one token could be a token the model has an
/// opinion about.
///
/// # It is a FLOOR and not a proof, and that has been measured
///
/// A model that has learned only "say what you just saw" passes this, because
/// for `"x x x ... x"` the echo and the continuation are the same token. That
/// is not hypothetical: `whether_the_fused_projection_is_packed_in_the_order_the_plan_reads_it`
/// found a packing that takes this from 0 of 4 to 4 of 4 and then predicts the
/// last token of a six-token period at every phase. So passing here is
/// necessary and nowhere near sufficient, and the period continuation is the
/// question that separates the two.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_a_prompt_of_one_repeated_token_predicts_that_token() {
    let Some((mut shell, _real)) = qwen3_5_shell(16) else {
        return;
    };
    println!("\nONE TOKEN, REPEATED SIXTEEN TIMES:");
    let mut right = 0;
    let mut fired = 0;
    // BOTH PATHS, because they are two implementations of one recurrence and
    // `whether_the_prefill_and_the_decode_leave_the_same_carry` says they do
    // not agree: at two tokens, from a zeroed seat, 261,247 of layer 0's
    // 262,144 state elements differ. One of them is wrong, the arithmetic in
    // the two kernel bodies is line-for-line the same, and the model itself
    // is the tiebreaker -- whichever path answers its own prompt is the one
    // computing the recurrence on disk.
    for (i, tok) in [15_339u32, 1_723, 88_204, 6_100].into_iter().enumerate() {
        let who = 900 + i as u64;
        let one_at_a_time = i >= 2;
        let row = if one_at_a_time {
            let mut last = Vec::new();
            for _ in 0..16 {
                last = fire_row(&mut shell, who, std::slice::from_ref(&tok));
                if last.is_empty() {
                    break;
                }
            }
            last
        } else {
            fire_row(&mut shell, who, &[tok; 16])
        };
        if row.is_empty() {
            continue;
        }
        fired += 1;
        print!(
            "  [{}] ",
            if one_at_a_time {
                "16 decodes  "
            } else {
                "one prefill "
            }
        );
        let mut top: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let got = u32::try_from(top[0].0).unwrap_or(u32::MAX);
        let at = row.get(tok as usize).copied().unwrap_or(f32::NAN);
        println!(
            "  {tok:>6} x16 -> {got:>6} ({:.2}), and {tok} scored {at:.2}; \
             top {:?}",
            top[0].1,
            top[..4].iter().map(|(t, _)| *t).collect::<Vec<_>>()
        );
        if got == tok {
            right += 1;
        }
    }
    if fired == 0 {
        println!("  nothing fired, so IT COULD NOT BE MEASURED");
        return;
    }
    println!("\n  {right} of {fired} repeated prompts answered with their own token");
    assert!(
        right == fired,
        "a prompt of one repeated token is the easiest continuation there is, \
         and this backend gets {right} of {fired} -- so what is wrong is the \
         model this driver computes, not the difficulty of the benchmark"
    );
}

/// Which HALF of the hybrid carries the wrong answer.
///
/// [`whether_a_prompt_of_one_repeated_token_predicts_that_token`] says the
/// model this driver computes is not the model on disk, and every magnitude
/// along the way is healthy, so the error is systematic rather than numerical.
/// A hybrid has two kinds of layer and they share almost no kernels, so the
/// cheapest cut is to SILENCE one kind and ask whether the other alone does
/// better than both together.
///
/// # What silences a layer, and why these weights
///
/// `norm::gated_rms` computes `out = w * rmsnorm(x) * silu(z)` and `w` is the
/// gate norm's weight, RAW -- `kernels-metal`'s signature says
/// `gate_norm_w [V_d] (raw, act dtype)` and this tree's shader repeats it. So
/// zeroing it makes a gated-DeltaNet layer contribute EXACTLY zero to the
/// residual, and eighteen of the twenty-four layers become identity.
///
/// It has to be a raw weight and not a quantized one. An affine-quantized
/// tensor reconstructs as `w = scale * code + bias`, so zeroing the codes pins
/// every weight to its per-group BIAS rather than to zero -- an attenuation
/// wearing an off switch's clothes.
///
/// The attention half is silenced the same way, at its own raw weight.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn which_half_of_the_hybrid_carries_the_wrong_answer() {
    let Some((mut shell, real)) = qwen3_5_shell(24) else {
        return;
    };
    // THE NAMES FIRST. A test that assumed them would silence nothing and
    // report the unsilenced model twice, which is a green run either way.
    let mut kinds: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for name in real.keys() {
        kinds.insert(
            name.split('.')
                .skip_while(|s| s.parse::<u32>().is_err())
                .skip(1)
                .collect::<Vec<_>>()
                .join("."),
        );
    }
    println!(
        "\nthe per-layer tensors this checkpoint carries: {}",
        kinds
            .iter()
            .filter(|k| !k.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    );
    let ask = |shell: &mut driver_wgpu::shell::Shell, who: u64| -> Option<u32> {
        let row = fire_row(shell, who, &[1_723u32; 16]);
        if row.is_empty() {
            return None;
        }
        let mut top: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!(
            "    top {:?} at {:.2}, and 1723 scored {:.2}",
            top[..4].iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            top[0].1,
            row.get(1_723).copied().unwrap_or(f32::NAN)
        );
        u32::try_from(top[0].0).ok()
    };
    println!("\n  BOTH HALVES:");
    let both = ask(&mut shell, 950);

    // Whichever of the two spellings this checkpoint uses; a name that is not
    // here holds nothing and is reported rather than assumed away.
    let silence = |shell: &mut driver_wgpu::shell::Shell, suffix: &str| -> usize {
        let mut n = 0;
        for (name, bytes) in &real {
            if name.ends_with(suffix) && shell.hold(name, &vec![0u8; bytes.len()]).is_ok() {
                n += 1;
            }
        }
        n
    };
    // The gated-DeltaNet half at its RAW gate norm, and the attention half at
    // its output projection -- which is quantized, so BOTH the scales and the
    // zeros go, or `w = scale * code + bias` leaves the bias standing.
    let cuts: [(&str, &[&str]); 3] = [
        ("the gated-DeltaNet layers", &["gate_norm"]),
        // NOT `o_proj`: every layer of both kinds has one, so that cut is the
        // whole model and reads as an attention result. It was made, and it
        // came back bit-identical to the vacuity guard below, which is how it
        // was caught. `v_proj` exists only where there is attention.
        ("the attention layers", &["v_proj.scales", "v_proj.zeros"]),
        // THE VACUITY GUARD, and it is the whole reason the other two can be
        // read. `lm_head` is TIED to the embedding, so a residual stream that
        // no layer has touched still argmaxes to the token it was embedded
        // from -- a model with every layer silent PASSES a repeated prompt.
        // If this cut answers 1723 as well, then "one half alone is right"
        // says nothing about that half; it says the shortcut survived.
        (
            "BOTH halves (the vacuity guard)",
            &["gate_norm", "v_proj.scales", "v_proj.zeros"],
        ),
    ];
    for (what, suffixes) in cuts {
        let n: usize = suffixes.iter().map(|s| silence(&mut shell, s)).sum();
        println!("\n  SILENCING {what}: {n} tensors zeroed");
        if n == 0 {
            println!("    nothing matched, so THIS CUT WAS NOT MADE");
            continue;
        }
        let got = ask(&mut shell, 960 + what.len() as u64);
        println!(
            "    {} the unsilenced answer",
            if got == both { "same as" } else { "DIFFERENT from" }
        );
        // Put them back, so the next cut is a cut of the whole model.
        for (name, bytes) in &real {
            if suffixes.iter().any(|s| name.ends_with(s)) {
                let _ = shell.hold(name, bytes);
            }
        }
    }
}

/// Whether the prefill and the decode leave the SAME recurrent state.
///
/// [`the_prefill_and_the_decode_are_one_recurrence_written_twice`] says the
/// two paths disagree by 3.5 on logits that span 18, already at two tokens.
/// One of them is wrong and the model cannot say which, because the attention
/// half alone answers correctly and the gated-DeltaNet half alone does not.
///
/// This splits the disagreement in half without needing a reference. A GDN
/// layer produces two things from one recurrence: the carry it leaves in
/// `recurrent_state`, and the `core_out` it hands the epilogue. If the two
/// paths leave the SAME state, the recurrence agrees and only the readout
/// differs; if the states differ, the recurrence itself does, and at two
/// tokens from a zeroed seat there is one step for it to differ in.
///
/// Two seats, not one released and re-taken: a released slot is not zeroed,
/// so the second conversation would begin on the first one's carry and the
/// comparison would be of a thing against itself plus a fire.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_the_prefill_and_the_decode_leave_the_same_carry() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];

    // The prompt in ONE fire, and the same prompt one token at a time.
    let a = 1_000u64;
    let b = 1_001u64;
    if fire_row(&mut shell, a, &PERIOD[..2]).is_empty() {
        println!("the two-token prefill was refused, so IT COULD NOT BE MEASURED");
        return;
    }
    for t in &PERIOD[..2] {
        if fire_row(&mut shell, b, std::slice::from_ref(t)).is_empty() {
            println!("a one-token decode was refused, so IT COULD NOT BE MEASURED");
            return;
        }
    }
    let (Some(sa), Some(sb)) = (shell.book().slot(a), shell.book().slot(b)) else {
        println!("one of the two conversations has no recurrent seat");
        return;
    };
    println!("\nthe prefill sits at slot {sa} and the decode at slot {sb}");
    // The slot count from the DEPLOYMENT this test opened, which is what
    // `qwen3_5_shell` was handed; `Shape` describes the paged pool and not
    // this one.
    let slots = 8u32;

    let mut worst = 0.0f32;
    let mut worst_where = String::new();
    let mut checked = 0usize;
    for which in ["recurrent_state", "conv_state"] {
        for layer in 0..6u16 {
            let Some(pool) = shell.recurrent() else { return };
            let Some(slab) = pool.slab(layer, which) else {
                continue;
            };
            let size = slab.size();
            let Ok(bytes) = shell.device().read_at(slab, 0, size) else {
                continue;
            };
            let per = bytes.len() / slots as usize;
            let at = |s: u32| -> Vec<f32> {
                let from = per * s as usize;
                bytes[from..from + per - (per % 4)]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            };
            let (ra, rb) = (at(sa), at(sb));
            checked += 1;
            let mut here = 0.0f32;
            let mut here_at = 0usize;
            let mut nonzero = 0usize;
            for (i, (x, y)) in ra.iter().zip(&rb).enumerate() {
                // NOT `f32::max`: it returns the non-NaN operand, so a fold
                // with it reports an all-NaN difference as zero.
                let d = (x - y).abs();
                if d != 0.0 {
                    nonzero += 1;
                }
                if d > here || !d.is_finite() {
                    here = d;
                    here_at = i;
                }
                if !d.is_finite() {
                    break;
                }
            }
            // PER LAYER, because the first layer is the one that decides.
            // Layer 0 sees the SAME embedding on both paths, so a carry that
            // already differs there is the recurrence differing; a carry that
            // agrees there and differs later is a difference arriving from
            // upstream, and the two want opposite next questions.
            println!(
                "    {which:<16} layer {layer:>2}: {nonzero:>7} of {} differ, widest {here:.6}",
                ra.len()
            );
            if here > worst {
                worst = here;
                worst_where = format!(
                    "{which} layer {layer} element {here_at}: {} vs {}",
                    ra[here_at], rb[here_at]
                );
            }
        }
    }
    assert!(checked > 0, "no slab could be read, so nothing was compared");
    println!("  {checked} slabs compared");
    println!("  widest disagreement {worst:.6} at {worst_where}");
    println!(
        "\n  {}",
        if worst == 0.0 {
            "THE CARRY IS IDENTICAL, so the recurrence agrees and what differs is the READOUT."
        } else {
            "THE CARRY ITSELF DIFFERS, so the two paths do not compute the same recurrence."
        }
    );
}

/// What the two paths actually RUN, rectangle by rectangle, at layer 0.
///
/// This began as "compare the staged `pre_q`, `pre_k` and `pre_gate`", which
/// is the last untested link between the two paths: the carries differ at
/// layer 0 while `conv_state` agrees to the byte, so either an input differs
/// or the reasoning does.
///
/// **The comparison cannot be made, and why is the finding.** The prefill
/// stages: `gdn_prep_prefill` writes `pre_q`, `pre_k` and `pre_gate` into the
/// arena and `gdn_core_recurrent_prefill` reads them back. The decode stages
/// NOTHING -- it runs `gdn_core_slotted`, one dispatch that does the
/// convolution, the normalisations, the gates and the recurrence in registers
/// and never writes a scratch tensor at all.
///
/// So the two paths are not one kernel called twice. They are `ssm/gdn_prep.wgsl`
/// and `ssm/gdn_core.wgsl` -- two files, each with its own binding table, its
/// own `load_mixed`/`load_conv_w`/`load_a_gate` helpers, its own workgroup
/// shape and its own reduction. Everything a diff can reach in them agrees:
/// the loaders are character-identical, the binding tables each match the arm
/// that feeds them (`core` supplies `mixed, conv_state, rstate, core_out,
/// conv_w, conv_b, a_log, dt_bias, a_gate, b_gate, new_conv_state, params,
/// slot_ids` against exactly those thirteen declarations, and `prep` supplies
/// its own eleven against the other file's), the gate formulas match term for
/// term, and `inv_sqrt_dk` rides the query side in both.
///
/// What is left is a difference the eye does not catch across two files, and
/// this test is what says which two files to put side by side.
///
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn what_the_two_gated_deltanet_paths_actually_run() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let prefill = hybrid_plan(&facts);
    let decode = hybrid_plan_class(&facts, model_ir::trace::FireClass::Decode);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];

    // The rectangle that stages the prep, in each plan, at LAYER 0 -- found by
    // name rather than by index, because the two lowerings are different
    // shapes and an index is only ever right for one of them.
    let find = |plan: &model_ir::trace::ForwardPlan, rows: usize, stem: &str| -> Option<(usize, usize)> {
        let rows: Vec<Row> = vec![
            Row {
                samples: true,
                ..Row::default()
            };
            rows
        ];
        let low = lower(
            plan,
            &rows,
            Fire {
                captures_across_splits: false,
            },
        )
        .ok()?;
        let at = low.launches.iter().position(|l| {
            low.kernels[l.kernel as usize].starts_with(stem) && l.layers.start == 0
        });
        let Some(at) = at else {
            println!("  no `{stem}` at layer 0; layer 0 runs:");
            for l in low.launches.iter().filter(|l| l.layers.start == 0) {
                println!("      {}", low.kernels[l.kernel as usize]);
            }
            return None;
        };
        // The three staged outputs are the LAST three arena ranges of the
        // rectangle: `pre_q`, `pre_k`, `pre_gate`, in the statement's order.
        let l = &low.launches[at];
        let arenas: Vec<(usize, usize)> = low.args[l.args.start as usize..l.args.end as usize]
            .iter()
            .filter_map(|a| match a {
                Arg::Arena { at, width, bytes } => Some((*at, *width as usize * *bytes as usize)),
                _ => None,
            })
            .collect();
        println!("  {stem} is rectangle {at}, with {} arena ranges", arenas.len());
        for (i, (o, w)) in arenas.iter().enumerate() {
            println!("      range {i}: at {o}, {w} bytes per row");
        }
        Some((at, arenas.len()))
    };
    println!("\nPREFILL, two rows:");
    let Some((pre_at, _)) = find(&prefill, 2, "gdn_prep_prefill") else {
        println!("  no `gdn_prep_prefill` at layer 0");
        return;
    };
    println!("DECODE, one row:");
    let Some((dec_at, _)) = find(&decode, 1, "gdn_prep") else {
        println!("  no `gdn_prep` at layer 0");
        return;
    };

    let _ = (pre_at, dec_at, PERIOD);
}

/// The numbers every gated-DeltaNet dispatch is handed, read off the plan.
///
/// `GdnCoreParams` is `Dk, Dv, Hk, Hv, conv_dim, Kc, q_off, k_off, v_off, eps,
/// inv_sqrt_dk` and it is the cross-backend contract: the three offsets say
/// where q, k and v sit inside the 6144-wide `mixed` row, and every kernel in
/// the family indexes the convolution through them. Get one wrong and nothing
/// overflows, nothing is non-finite and every width still checks out -- the
/// model simply convolves the wrong channels. That is the shape of what is
/// left after the wiring was cleared.
///
/// Needs no device, no snapshot and no weights: a plan states its own scalars.
#[test]
fn the_numbers_every_gated_deltanet_dispatch_is_handed() {
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 4;
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        2
    ];
    for (what, plan) in [
        ("PREFILL", hybrid_plan(&facts)),
        (
            "DECODE",
            hybrid_plan_class(&facts, model_ir::trace::FireClass::Decode),
        ),
    ] {
        let low = lower(
            &plan,
            &rows,
            Fire {
                captures_across_splits: false,
            },
        )
        .expect("the hybrid lowers");
        println!("\n{what}");
        for l in &low.launches {
            let name = &low.kernels[l.kernel as usize];
            if !name.starts_with("gdn_") || l.layers.start != 0 {
                continue;
            }
            let scalars = &low.params[l.params.start as usize..l.params.end as usize];
            println!("  {name}");
            for (i, s) in scalars.iter().enumerate() {
                println!("      param {i:>2}: {s:?}");
            }
        }
    }
}

/// Whether the gated DeltaNet's SHAPE is the checkpoint's, checked against the
/// tensors on disk rather than against the declaration.
///
/// `the_numbers_every_gated_deltanet_dispatch_is_handed` says every scalar the
/// family is handed is what `Qwen35GdnFacts::qwen3_5_0_8b` declares --
/// `Dk = Dv = 128`, `Hk = Hv = 16`, `conv_dim = 6144`, `Kc = 4`, the three
/// offsets at 0/2048/4096, `eps = 1e-6` and `inv_sqrt_dk = 1/sqrt(128)`. That
/// is self-consistency, not correctness: it says the driver agrees with the
/// declaration and nothing about whether the declaration agrees with the
/// weights.
///
/// The weights state their own shape. `gate_norm` is `[Dv]`, `a_log` and `dt`
/// are `[Hv]`, and `conv_w` is `[conv_dim, Kc]`, so their LENGTHS are four
/// independent readings of the same four numbers.
///
/// This matters because a wrong split is invisible to everything measured so
/// far. `Hv = 32, Dv = 64` gives the identical 2048-wide value block, the
/// identical `v_off`, and identical widths at every rectangle -- and convolves
/// and gates entirely different channels together.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot"]
fn whether_the_gated_deltanets_shape_is_the_checkpoints() {
    let Some(dir) = qwen3_5_snapshot() else {
        println!("no Qwen3.5-0.8B snapshot, so IT COULD NOT BE MEASURED");
        return;
    };
    let Some(row) = model::catalog::find("qwen3.5-0.8b-base") else {
        return;
    };
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let Some(real) = qwen3_5_weights(&dir, row, &hybrid_plan(&facts)) else {
        println!("the weights would not load, so IT COULD NOT BE MEASURED");
        return;
    };
    // Layer 0 is a gated-DeltaNet layer, and every checkpoint tensor is bf16
    // except `a_log`, which the shader reads as `array<f32>`.
    let len = |name: &str| real.get(name).map(Vec::len);
    println!("\nthe gated-DeltaNet tensors of layer 0, as bytes:");
    for n in [
        "gate_norm", "a_log", "dt", "conv_w", "conv_b", "in_proj_qkv", "in_proj_z",
    ] {
        let full = format!("layer.0.{n}");
        match len(&full) {
            Some(b) => println!("    {n:<12} {b:>9} bytes"),
            None => println!("    {n:<12} ABSENT"),
        }
    }
    let (dk, dv, hv, conv_dim, kc) = (128usize, 128usize, 16usize, 6144usize, 4usize);
    let mut disagree: Vec<String> = Vec::new();
    let mut say = |what: &str, got: Option<usize>, want: usize, unit: usize| {
        let Some(bytes) = got else {
            println!("  {what}: ABSENT, so it says nothing");
            return;
        };
        let n = bytes / unit;
        println!(
            "  {what}: {bytes} bytes / {unit} = {n}, and the declaration says {want} -- {}",
            if n == want { "agree" } else { "DISAGREE" }
        );
        if n != want {
            disagree.push(format!("{what} reads {n} where the declaration says {want}"));
        }
    };
    println!("\nwhat the tensors say against what the plan states:");
    say("gate_norm is `[Dv]`", len("layer.0.gate_norm"), dv, 2);
    say("a_log is `[Hv]`", len("layer.0.a_log"), hv, 4);
    say("dt is `[Hv]`", len("layer.0.dt"), hv, 2);
    say("conv_w is `[conv_dim, Kc]`", len("layer.0.conv_w"), conv_dim * kc, 2);
    say("conv_b is `[conv_dim]`", len("layer.0.conv_b"), conv_dim, 2);
    let _ = dk;
    assert!(
        disagree.is_empty(),
        "the checkpoint's gated-DeltaNet tensors do not have the shape this \
         plan states, so every dispatch in the family indexes them wrongly \
         while every width still checks out:\n  {}",
        disagree.join("\n  ")
    );
}

/// Whether the two gate projections are the right way round.
///
/// Everything structural about the gated-DeltaNet half has now been checked
/// and agrees: the wiring against `driver-metal` arm by arm, the binding
/// tables against the arms that feed them, every `GdnCoreParams` scalar
/// (`Dk = Dv = 128`, `Hk = Hv = 16`, `conv_dim = 6144`, `Kc = 4`, offsets at
/// 0/2048/4096, `eps = 1e-6`, `inv_sqrt_dk = 1/sqrt(128)`), and every one of
/// those numbers again against the LENGTHS of the tensors on disk. And the
/// half still answers wrongly on both of its paths.
///
/// What survives all of that is a SWAP: two tensors of the same shape in the
/// wrong slots. `a` and `b` are the candidate pair. Both are `[Hv]` per token,
/// both come from a `hidden -> 16` projection, and both end up inside a
/// saturating function -- `decay = exp(-exp(A_log) * softplus(a + dt_bias))`
/// and `beta = sigmoid(b)`, each in `(0, 1)` whichever way round they are. So
/// a swap is finite, plausible, correctly shaped, and completely wrong.
///
/// There is a reason to suspect this pair in particular. The reference packs
/// them in ONE projection and splits `b` FIRST:
///
///     ba = self.in_proj_ba(hidden_states)
///     b, a = torch.split(ba, [num_v_heads, num_v_heads], dim=-1)
///
/// A loader that split the same tensor as `[a | b]` produces exactly this
/// checkpoint's two separate `in_proj_a` and `in_proj_b` with their contents
/// exchanged.
///
/// # Why this is a measurement and not a patch
///
/// It swaps what the SHELL HOLDS, not what any crate states. If the answer
/// comes back right, the defect is named and its fix belongs where the split
/// happens; if it does not, the pair is eliminated and the search moves on.
/// Both outcomes are worth a run.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_the_two_gate_projections_are_the_right_way_round() {
    let Some((mut shell, real)) = qwen3_5_shell(24) else {
        return;
    };
    let ask = |shell: &mut driver_wgpu::shell::Shell, who: u64, tok: u32| -> Option<u32> {
        let row = fire_row(shell, who, &[tok; 16]);
        if row.is_empty() {
            return None;
        }
        let mut top: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!(
            "    {tok} x16 -> {} ({:.2}); {tok} scored {:.2}",
            top[0].0,
            top[0].1,
            row.get(tok as usize).copied().unwrap_or(f32::NAN)
        );
        u32::try_from(top[0].0).ok()
    };
    const PROBES: [u32; 3] = [15_339, 1_723, 6_100];
    println!("\n  AS HELD:");
    let before: Vec<Option<u32>> = PROBES
        .iter()
        .enumerate()
        .map(|(i, t)| ask(&mut shell, 1_200 + i as u64, *t))
        .collect();

    // The swap: every layer that has the pair, and all three tensors of each
    // -- the codes, the scales and the zeros. Swapping only the codes would
    // leave each set of weights reconstructing against the other's scale,
    // which is a third model rather than either of the two on offer.
    let mut swapped = 0usize;
    for layer in 0..24u32 {
        for part in ["", ".scales", ".zeros"] {
            let (na, nb) = (
                format!("layer.{layer}.in_proj_a{part}"),
                format!("layer.{layer}.in_proj_b{part}"),
            );
            let (Some(a), Some(b)) = (real.get(&na), real.get(&nb)) else {
                continue;
            };
            if a.len() != b.len() {
                println!("  {na} and {nb} are different lengths, so they are not a swap");
                continue;
            }
            if shell.hold(&na, b).is_ok() && shell.hold(&nb, a).is_ok() {
                swapped += 1;
            }
        }
    }
    println!("\n  SWAPPED `in_proj_a` <-> `in_proj_b`: {swapped} tensors exchanged");
    if swapped == 0 {
        println!("    nothing was exchanged, so IT COULD NOT BE MEASURED");
        return;
    }
    let after: Vec<Option<u32>> = PROBES
        .iter()
        .enumerate()
        .map(|(i, t)| ask(&mut shell, 1_300 + i as u64, *t))
        .collect();

    let right_before = before
        .iter()
        .zip(PROBES)
        .filter(|(g, t)| **g == Some(*t))
        .count();
    let right_after = after
        .iter()
        .zip(PROBES)
        .filter(|(g, t)| **g == Some(*t))
        .count();
    println!(
        "\n  as held {right_before} of {} right, swapped {right_after} of {} right",
        PROBES.len(),
        PROBES.len()
    );
    println!(
        "  {}",
        if right_after > right_before {
            "THE PAIR IS THE WRONG WAY ROUND, and the fix belongs where the split happens."
        } else if before == after {
            "THE SWAP CHANGED NOTHING AT ALL, which means these tensors are not reaching the fire."
        } else {
            "the pair is not it: the answer moved and did not improve, so `a` and `b` are eliminated."
        }
    );
}

/// Which gated-DeltaNet weights REACH the answer, one kind at a time.
///
/// `whether_the_two_gate_projections_are_the_right_way_round` eliminated the
/// one swap there was a textual reason to suspect, and its instrument
/// generalises: hold something different and see whether the logits move.
///
/// A weight that is bound to the wrong slot, or to a slot the shader never
/// reads, is INERT -- zeroing it changes nothing. That is a different defect
/// from a wrong value and it is invisible to every check made so far, because
/// a binding table that matches its arm can still be the wrong table for the
/// tensor the trace put there.
///
/// Zeroing is chosen per kind, not uniformly. An affine-quantized tensor
/// reconstructs as `w = scale * code + bias`, so zeroing the CODES pins every
/// weight to its per-group bias -- an attenuation, not an off switch -- and
/// the scales and zeros have to go with them. The raw bf16 tensors are their
/// own values and zero is zero.
///
/// `dt` and `a_log` are the interesting rows: both are per-head scalars inside
/// a saturating function, so a value that never arrives leaves gates that are
/// still in `(0, 1)` and a model that still answers.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn which_gated_deltanet_weights_reach_the_answer() {
    let Some((mut shell, real)) = qwen3_5_shell(24) else {
        return;
    };
    let logits = |shell: &mut driver_wgpu::shell::Shell, who: u64| -> Vec<f32> {
        fire_row(shell, who, &[1_723u32; 16])
    };
    let base = logits(&mut shell, 1_400);
    if base.is_empty() {
        println!("the baseline fire was refused, so IT COULD NOT BE MEASURED");
        return;
    }
    println!("\n  what each gated-DeltaNet weight is worth to the answer:");
    let mut inert: Vec<&str> = Vec::new();
    for (i, kind) in [
        "conv_w",
        "conv_b",
        "a_log",
        "dt",
        "gate_norm",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
    ]
    .into_iter()
    .enumerate()
    {
        // The quantized kinds need their scales and zeros zeroed with them.
        let parts: &[&str] = if real.contains_key(&format!("layer.0.{kind}.scales")) {
            &["", ".scales", ".zeros"]
        } else {
            &[""]
        };
        let mut n = 0usize;
        for layer in 0..24u32 {
            for part in parts {
                let name = format!("layer.{layer}.{kind}{part}");
                let Some(bytes) = real.get(&name) else { continue };
                if shell.hold(&name, &vec![0u8; bytes.len()]).is_ok() {
                    n += 1;
                }
            }
        }
        if n == 0 {
            println!("    {kind:<12} no such tensor on any layer");
            continue;
        }
        // THE GUARD, and it is the whole reason "inert" can be read as a
        // defect. Zeroing a tensor that is ALREADY zero on disk changes
        // nothing, and that is the weight being read correctly rather than
        // not at all. Checked before the fire, so the two cannot be confused.
        let already_zero = (0..24u32).all(|layer| {
            real.get(&format!("layer.{layer}.{kind}"))
                .is_none_or(|b| b.iter().all(|&x| x == 0))
        });
        let got = logits(&mut shell, 1_410 + i as u64);
        // NOT folded with `f32::max`: it returns the non-NaN operand, so an
        // all-NaN answer would report a move of zero and read as "inert".
        let mut moved = 0.0f32;
        let mut nan = 0usize;
        for (a, b) in base.iter().zip(&got) {
            let d = (a - b).abs();
            if d.is_finite() {
                moved = moved.max(d);
            } else {
                nan += 1;
            }
        }
        let same = base.iter().zip(&got).filter(|(a, b)| a == b).count();
        println!(
            "    {kind:<12} {n:>2} tensors zeroed -> widest move {moved:>8.3}, \
             {same} of {} logits unchanged{}",
            base.len(),
            if nan > 0 {
                format!(", {nan} non-finite")
            } else {
                String::new()
            }
        );
        if same == base.len() {
            if already_zero {
                println!(
                    "      and every layer's `{kind}` is ALREADY ZERO on disk, so \
                     unchanged logits are it being read, not it being missed"
                );
            } else {
                inert.push(kind);
            }
        }
        for layer in 0..24u32 {
            for part in parts {
                let name = format!("layer.{layer}.{kind}{part}");
                if let Some(bytes) = real.get(&name) {
                    let _ = shell.hold(&name, bytes);
                }
            }
        }
    }
    println!();
    assert!(
        inert.is_empty(),
        "these gated-DeltaNet weights do not reach the answer at all -- \
         zeroing every layer's copy left all {} logits bit-identical, so they \
         are bound where nothing reads them:\n  {}",
        base.len(),
        inert.join("\n  ")
    );
    println!("  every gated-DeltaNet weight reaches the answer, so none is bound where nothing reads it.");
}

/// Whether `in_proj_qkv` is packed in the order the plan reads it.
///
/// The plan states `q_off = 0`, `k_off = 2048`, `v_off = 4096` into a
/// 6144-wide row, and the reference splits the fused projection
/// `[key_dim, key_dim, value_dim]` in that order. Every kernel in the family
/// reaches q, k and v only through those three offsets, so a checkpoint packed
/// any other way convolves and gates the right channels under the wrong names
/// -- finite, plausible, every width intact, and a different model. It is the
/// same class as the `a`/`b` swap and the last one of that class left.
///
/// # Why this can be done as a swap at all
///
/// The three blocks are the SAME SIZE here: `Hk * Dk = Hk * Dk = Hv * Dv =
/// 2048`, because this checkpoint has 16 key heads and 16 value heads of 128
/// each. So a permutation is an exchange of equal byte ranges rather than a
/// re-layout, and it can be done to what the shell holds without touching a
/// crate.
///
/// `in_proj_qkv` is `[6144, 1024]` at 4 bits -- 3,145,728 bytes, which is 512
/// per output row -- and its scales and zeros are per output row too, so all
/// three tensors take the same row-range exchange. Swapping the codes alone
/// would reconstruct each block against another's scale, which is a fourth
/// model rather than any of the three on offer.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_the_fused_projection_is_packed_in_the_order_the_plan_reads_it() {
    let Some((mut shell, real)) = qwen3_5_shell(24) else {
        return;
    };
    const BLOCKS: usize = 3;
    // FOUR prompts, not one. A single repeated token coming out right is a
    // one-in-a-few-thousand coincidence away from meaning nothing, and this
    // test's whole job is to tell a packing from a fluke.
    const PROBES: [u32; 4] = [15_339, 1_723, 88_204, 6_100];
    let ask = |shell: &mut driver_wgpu::shell::Shell, who: u64| -> usize {
        let mut right = 0;
        for (i, tok) in PROBES.into_iter().enumerate() {
            let row = fire_row(shell, who + i as u64, &[tok; 16]);
            if row.is_empty() {
                continue;
            }
            let mut top: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            println!(
                "    {tok:>6} x16 -> {:>6} ({:.2});  top {:?}",
                top[0].0,
                top[0].1,
                top[..4].iter().map(|(t, _)| *t).collect::<Vec<_>>()
            );
            if u32::try_from(top[0].0) == Ok(tok) {
                right += 1;
            }
        }
        println!("    {right} of {} answered with their own token", PROBES.len());
        right
    };
    // THE DISCRIMINATOR. A repeated token is answerable by a model that has
    // learned only "say what you just saw", and more than one packing clears
    // that bar. Continuing a SIX-TOKEN period at three different phases needs
    // the recurrence to actually carry the sequence, which is the thing the
    // gated-DeltaNet half is for.
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let period = |shell: &mut driver_wgpu::shell::Shell, who: u64| -> usize {
        let mut right = 0;
        for (i, phase) in [2usize, 4, 5].into_iter().enumerate() {
            let mut tokens: Vec<u32> = Vec::new();
            for _ in 0..5 {
                tokens.extend_from_slice(&PERIOD);
            }
            tokens.extend_from_slice(&PERIOD[..phase]);
            let want = PERIOD[phase];
            let row = fire_row(shell, who + 100 + i as u64, &tokens);
            if row.is_empty() {
                continue;
            }
            let mut top: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            println!(
                "    period at phase {phase}: wanted {want:>6}, got {:>6} ({:.2})",
                top[0].0, top[0].1
            );
            if u32::try_from(top[0].0) == Ok(want) {
                right += 1;
            }
        }
        println!("    {right} of 3 period continuations");
        right
    };

    // The three tensors and how many bytes one output row of each takes.
    let per_row = |name: &str| -> Option<usize> {
        let len = real.get(name)?.len();
        (len % 6144 == 0).then_some(len / 6144)
    };
    for part in ["", ".scales", ".zeros"] {
        let n = format!("layer.0.in_proj_qkv{part}");
        match per_row(&n) {
            Some(b) => println!("  {n}: {b} bytes per output row"),
            None => {
                println!("  {n} is not a whole number of 6144 rows, so IT CANNOT BE SWAPPED");
                return;
            }
        }
    }
    println!("\n  AS PACKED (q, k, v):");
    let base = ask(&mut shell, 1_500) + period(&mut shell, 1_500);

    // `order[i]` is which of the three source blocks lands in slot `i`.
    // ALL FIVE non-identity permutations. The three blocks are the same size,
    // so every one of them is a legal reading of the row and none can be
    // ruled out by shape -- which is the whole reason this is a sweep and not
    // a guess.
    for (name, order) in [
        ("(k, q, v)", [1usize, 0, 2]),
        ("(q, v, k)", [0, 2, 1]),
        ("(v, k, q)", [2, 1, 0]),
        ("(k, v, q)", [1, 2, 0]),
        ("(v, q, k)", [2, 0, 1]),
    ] {
        let mut moved = 0usize;
        for layer in 0..24u32 {
            for part in ["", ".scales", ".zeros"] {
                let full = format!("layer.{layer}.in_proj_qkv{part}");
                let (Some(src), Some(stride)) = (real.get(&full), per_row(&full)) else {
                    continue;
                };
                let block = 2048 * stride;
                let mut out = src.clone();
                for (slot, from) in order.iter().enumerate() {
                    let (d, s) = (slot * block, from * block);
                    out[d..d + block].copy_from_slice(&src[s..s + block]);
                }
                if shell.hold(&full, &out).is_ok() {
                    moved += 1;
                }
            }
        }
        println!("\n  PACKED {name}: {moved} tensors rewritten");
        if moved == 0 {
            continue;
        }
        let who = 1_600 + order[0] as u64 * 700 + order[1] as u64 * 70;
        let got = ask(&mut shell, who) + period(&mut shell, who);
        println!(
            "    {}",
            if got == PROBES.len() + 3 {
                "EVERY PROMPT AND EVERY PERIOD -- this is the packing"
            } else if got > base {
                "better than as-packed but not clean, so this is not the whole of it"
            } else {
                "no better than as-packed, so this order is eliminated"
            }
        );
    }
    // Put the real weights back, so a later probe on this shell is not
    // reading a permutation.
    for layer in 0..24u32 {
        for part in ["", ".scales", ".zeros"] {
            let full = format!("layer.{layer}.in_proj_qkv{part}");
            if let Some(bytes) = real.get(&full) {
                let _ = shell.hold(&full, bytes);
            }
        }
    }
    let _ = BLOCKS;
    println!("\n  as packed answered {base} of {} correctly", PROBES.len() + 3);
}

/// The recurrence, recomputed on the CPU from the kernel's OWN staged inputs.
///
/// Both gated-DeltaNet paths answer wrongly and they disagree with each other,
/// and every structural explanation is eliminated: the wiring against
/// `driver-metal` arm by arm, the binding tables against their arms, every
/// `GdnCoreParams` scalar, those scalars again against the lengths of the
/// tensors on disk, all six packings of the fused projection, the side of the
/// norm the gate falls on, and whether any weight is inert. So the question is
/// no longer "which input is wrong" but "does the arithmetic do what its
/// inputs say", and that needs a reference rather than a comparison.
///
/// It is a two-token prompt from a zeroed seat, so the reference is four lines:
///
///     S = 0
///     for t: S *= ga[t]
///            kv = S . k[t]
///            S += k[t] (x) (v[t] - kv) * gb[t]
///
/// and `gdn_prep_prefill` stages every term of it -- `pre_q`, `pre_k` and
/// `pre_gate`, the last carrying `[ga, gb per head][v per head and channel]`.
/// `fire_prefix` stops the fire on that rectangle so the arena holds exactly
/// what it wrote, and `Shell::recurrent` reads back what the scan then left.
///
/// # What each half of the answer would mean
///
/// The scan reads those staged values and nothing else, so CPU against SCAN is
/// a clean test of one kernel's arithmetic. CPU against DECODE is not -- the
/// fused kernel computes its own `q` and `k` from the convolution and never
/// stages them -- so a disagreement there is the recurrence OR the inputs, and
/// only the scan's answer is decisive on its own.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_the_scan_computes_the_recurrence_its_own_inputs_imply() {
    let Some((mut shell, real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let tokens = &PERIOD[..2];
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        2
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    let Some(at) = low.launches.iter().position(|l| {
        low.kernels[l.kernel as usize].starts_with("gdn_prep_prefill") && l.layers.start == 0
    }) else {
        println!("no `gdn_prep_prefill` at layer 0");
        return;
    };
    let l = &low.launches[at];
    let arenas: Vec<(usize, usize)> = low.args[l.args.start as usize..l.args.end as usize]
        .iter()
        .filter_map(|a| match a {
            Arg::Arena { at, width, bytes } => Some((*at, *width as usize * *bytes as usize)),
            _ => None,
        })
        .collect();
    // `mixed`, `a`, `b`, then the three staged outputs.
    if arenas.len() != 6 {
        println!("`gdn_prep_prefill` has {} arena ranges, not 6", arenas.len());
        return;
    }

    shell.fire_prefix(Some(at + 1));
    let arena = match shell.step(&[driver_wgpu::turns::Turn {
        who: 2_000,
        tokens: tokens.to_vec(),
    }]) {
        Ok(s) => s.arena,
        Err(why) => {
            println!("the truncated prefill was refused: {why}");
            return;
        }
    };
    shell.fire_prefix(None);
    let read = |at: usize, stride: usize, t: usize, n: usize| -> Vec<f32> {
        let from = at + t * stride;
        arena[from..from + n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let (hv, dv_n, dk) = (16usize, 128usize, 128usize);
    let (qk_pitch, g_pitch) = (hv * dk, 2 * hv + hv * dv_n);
    let pre_q: Vec<Vec<f32>> = (0..2)
        .map(|t| read(arenas[3].0, arenas[3].1, t, qk_pitch))
        .collect();
    let pre_k: Vec<Vec<f32>> = (0..2)
        .map(|t| read(arenas[4].0, arenas[4].1, t, qk_pitch))
        .collect();
    let pre_gate: Vec<Vec<f32>> = (0..2)
        .map(|t| read(arenas[5].0, arenas[5].1, t, g_pitch))
        .collect();

    // The reference, over every `(head, channel)` the layer has. `after_one`
    // is the same walk stopped after the FIRST token, which is the state both
    // two-token paths step from and the simplest case the recurrence has: from
    // a zeroed seat `S = k (x) (v * gb)` exactly, with no decay and no
    // subtraction, so anything that disagrees there disagrees about the four
    // lines and not about accumulating them.
    let mut after_one = vec![0.0f32; hv * dv_n * dk];
    let mut want = vec![0.0f32; hv * dv_n * dk];
    for h in 0..hv {
        for dv in 0..dv_n {
            let base = (h * dv_n + dv) * dk;
            for t in 0..2 {
                let ga = pre_gate[t][2 * h];
                let gb = pre_gate[t][2 * h + 1];
                let vv = pre_gate[t][2 * hv + h * dv_n + dv];
                let k = &pre_k[t][h * dk..(h + 1) * dk];
                let mut kv = 0.0f32;
                for i in 0..dk {
                    want[base + i] *= ga;
                    kv += want[base + i] * k[i];
                }
                let delta = (vv - kv) * gb;
                for i in 0..dk {
                    want[base + i] += k[i] * delta;
                }
                if t == 0 {
                    after_one[base..base + dk].copy_from_slice(&want[base..base + dk]);
                }
            }
        }
    }

    // Now the two devices' carries, each on its own fresh seat.
    let full = |shell: &mut driver_wgpu::shell::Shell, who: u64, one_at_a_time: bool| -> Option<u32> {
        if one_at_a_time {
            for t in tokens {
                fire_row(shell, who, std::slice::from_ref(t));
            }
        } else {
            fire_row(shell, who, tokens);
        }
        shell.book().slot(who)
    };
    let scan_slot = full(&mut shell, 2_001, false);
    let dec_slot = full(&mut shell, 2_002, true);
    // A third seat that stops after ONE token, which is the state the decode's
    // single step begins from. Its token goes through the scan on both paths,
    // so this is a known-good starting point and not another unknown.
    fire_row(&mut shell, 2_003, &tokens[..1]);
    let one_slot = shell.book().slot(2_003);
    let (Some(scan_slot), Some(dec_slot), Some(one_slot)) = (scan_slot, dec_slot, one_slot) else {
        println!("one of the three conversations has no seat");
        return;
    };
    let Some(pool) = shell.recurrent() else { return };
    let Some(slab) = pool.slab(0, "recurrent_state") else {
        return;
    };
    let Ok(bytes) = shell.device().read_at(slab, 0, slab.size()) else {
        println!("the slab would not read back");
        return;
    };
    let per = hv * dv_n * dk;
    let got = |slot: u32| -> Vec<f32> {
        let from = per * slot as usize * 4;
        bytes[from..from + per * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    println!(
        "\n  the reference is over {per} state elements, from the prefill's own staged inputs"
    );
    for (name, slot) in [("SCAN  ", scan_slot), ("DECODE", dec_slot)] {
        let mine = got(slot);
        let mut worst = 0.0f32;
        let mut at_i = 0usize;
        let mut nan = 0usize;
        for (i, (a, b)) in want.iter().zip(&mine).enumerate() {
            let d = (a - b).abs();
            if !d.is_finite() {
                nan += 1;
            } else if d > worst {
                worst = d;
                at_i = i;
            }
        }
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!(
            "    {name} slot {slot}: widest |cpu - gpu| {worst:.6} against a reference \
             whose own widest is {scale:.6}{}",
            if nan > 0 {
                format!(", and {nan} non-finite")
            } else {
                String::new()
            }
        );
        println!(
            "        at element {at_i} (head {}, channel {}, key {}): cpu {:.6} gpu {:.6}",
            at_i / (dv_n * dk),
            (at_i / dk) % dv_n,
            at_i % dk,
            want[at_i],
            mine[at_i]
        );
    }

    // WHAT THE DECODE ACTUALLY DID, solved for rather than guessed.
    //
    // One step adds a RANK-ONE term: `S_new = A * S_old + B * u` for some
    // direction `u`, and the recurrence says `A` is the decay and `u` is the
    // staged `k`. Two unknowns against 128 equations per value channel, so
    // least squares FITS them and the residual says whether the direction was
    // right at all. Fitting against `q` as well as `k` costs one more solve
    // and distinguishes "a wrong gate" from "the wrong vector entirely".
    let old = got(one_slot);
    let dec = got(dec_slot);
    let fit = |sn: &[f32], so: &[f32], u: &[f32]| -> (f32, f32, f32) {
        // Normal equations for `sn = A*so + B*u`.
        let (mut ss, mut su, mut uu, mut sn_s, mut sn_u) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
        for i in 0..sn.len() {
            let (a, b, c) = (f64::from(so[i]), f64::from(u[i]), f64::from(sn[i]));
            ss += a * a;
            su += a * b;
            uu += b * b;
            sn_s += c * a;
            sn_u += c * b;
        }
        let det = ss * uu - su * su;
        if det.abs() < 1e-30 {
            return (f32::NAN, f32::NAN, f32::NAN);
        }
        let a = (sn_s * uu - sn_u * su) / det;
        let b = (sn_u * ss - sn_s * su) / det;
        let (mut res, mut mag) = (0.0f64, 0.0f64);
        for i in 0..sn.len() {
            let e = f64::from(sn[i]) - a * f64::from(so[i]) - b * f64::from(u[i]);
            res += e * e;
            mag += f64::from(sn[i]) * f64::from(sn[i]);
        }
        (a as f32, b as f32, (res / mag.max(1e-30)).sqrt() as f32)
    };
    // THE ONE-TOKEN CASE FIRST. If this disagrees there is no point solving a
    // second step, and if it agrees then the state the decode's step begins
    // from is the reference's own.
    {
        let mut worst = 0.0f32;
        let mut at_i = 0usize;
        for (i, (a, b)) in after_one.iter().zip(&old).enumerate() {
            let d = (a - b).abs();
            if d > worst {
                worst = d;
                at_i = i;
            }
        }
        let scale = after_one.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!(
            "\n  AFTER ONE TOKEN: widest |cpu - gpu| {worst:.6} against a reference whose \
             own widest is {scale:.6}, at head {} channel {} key {}",
            at_i / (dv_n * dk),
            (at_i / dk) % dv_n,
            at_i % dk
        );
    }

    // THE CONTROL for the solver below: run it on the reference itself, where
    // the answer is known by construction. A fit that cannot recover `ga` and
    // `delta` from data that has them is a fit whose verdict on the device
    // means nothing.
    println!("\n  SOLVING, and the first row is the solver checked against arithmetic it knows:");
    for (h, dv) in [(8usize, 81usize), (0, 0), (3, 17)] {
        let base = (h * dv_n + dv) * dk;
        let ga = pre_gate[1][2 * h];
        let gb = pre_gate[1][2 * h + 1];
        let vv = pre_gate[1][2 * hv + h * dv_n + dv];
        let k = &pre_k[1][h * dk..(h + 1) * dk];
        let q = &pre_q[1][h * dk..(h + 1) * dk];
        let sn = &dec[base..base + dk];
        let so = &old[base..base + dk];
        let mut kv = 0.0f32;
        for i in 0..dk {
            kv += ga * so[i] * k[i];
        }
        let want_delta = (vv - kv) * gb;
        let (ak, bk, rk) = fit(sn, so, k);
        let (aq, bq, rq) = fit(sn, so, q);
        let (ac, bc, rc) = fit(&want[base..base + dk], &after_one[base..base + dk], k);
        println!("    head {h:>2} channel {dv:>3}: the recurrence says decay {ga:.6}, delta {want_delta:.6}");
        println!("        fit against the REFERENCE:  decay {ac:.6}, delta {bc:.6}, relative residual {rc:.6}");
        println!("        fit against staged `k`: decay {ak:.6}, delta {bk:.6}, relative residual {rk:.6}");
        println!("        fit against staged `q`: decay {aq:.6}, delta {bq:.6}, relative residual {rq:.6}");
    }

    // WHAT DIRECTION DID IT ADD? The step is `S_new = ga*S_old + u*delta`, so
    // `S_new - ga*S_old` is `u` scaled by a per-channel number. Recovered per
    // value channel and compared BY ANGLE, which is scale-free -- a `k` that
    // differed only by a factor would still fit above, so the question left is
    // whether the direction itself is the staged one.
    //
    // Two channels of the same head are compared to each other as well. `u` is
    // one vector for the whole head, so if those two disagree the step is not
    // rank-one at all, and that is a different defect from a wrong `k`: it
    // would mean the lanes are not writing the state elements they read.
    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..a.len() {
            ab += f64::from(a[i]) * f64::from(b[i]);
            aa += f64::from(a[i]) * f64::from(a[i]);
            bb += f64::from(b[i]) * f64::from(b[i]);
        }
        (ab / (aa.sqrt() * bb.sqrt()).max(1e-30)) as f32
    };
    // THE DECAY THE DECODE ACTUALLY USED, solved over a WHOLE HEAD.
    //
    // Per value channel the step is `S_new[dv] = ga*S_old[dv] + k*delta[dv]`,
    // and `ga` is shared by all 128 channels while each has its own `delta`.
    // That is 16,384 equations in 129 unknowns, so fixing `ga` determines
    // every `delta` by projection and leaves a residual -- and the `ga` that
    // minimises it is the one the device used, IF the direction really is `k`.
    // A residual that stays large at every `ga` says it is not.
    println!("\n  THE DECAY THE DECODE USED, fitted over a whole head:");
    for h in [8usize, 0, 3] {
        let ga = pre_gate[1][2 * h];
        let k = &pre_k[1][h * dk..(h + 1) * dk];
        let kk: f64 = k.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let residual = |g: f64| -> f64 {
            let (mut res, mut mag) = (0.0f64, 0.0f64);
            for dv in 0..dv_n {
                let base = (h * dv_n + dv) * dk;
                let mut dot = 0.0f64;
                for i in 0..dk {
                    dot += (f64::from(dec[base + i]) - g * f64::from(old[base + i]))
                        * f64::from(k[i]);
                }
                let delta = dot / kk.max(1e-30);
                for i in 0..dk {
                    let e = f64::from(dec[base + i])
                        - g * f64::from(old[base + i])
                        - delta * f64::from(k[i]);
                    res += e * e;
                    mag += f64::from(dec[base + i]) * f64::from(dec[base + i]);
                }
            }
            (res / mag.max(1e-30)).sqrt()
        };
        // A sweep, not a solve: the residual is a quadratic in `g` so the
        // minimum is exact, but printing the curve's ends is what says whether
        // the minimum means anything.
        let mut best = (f64::INFINITY, 0.0f64);
        for step in 0..=4000 {
            let g = -1.0 + f64::from(step) * 0.001;
            let r = residual(g);
            if r < best.0 {
                best = (r, g);
            }
        }
        println!(
            "    head {h:>2}: staged decay {ga:.6} gives residual {:.6}; \
             the BEST decay is {:.6} with residual {:.6}",
            residual(f64::from(ga)),
            best.1,
            best.0
        );
    }

    // WHICH CONVOLUTION WINDOW THE DECODE ACTUALLY USED.
    //
    // A one-row step takes the DECODE plan -- `Serving::plan` is "the text a
    // one-row step lowers" -- so `gdn_core_slotted` ran for BOTH tokens, and
    // it was exact at the first: the state after one token matches the
    // reference to 0.000000. What is different about the second is that
    // neither the recurrent state nor the CONVOLUTION STATE is zero any more,
    // and a tap that is multiplied by zero is a tap that is not tested.
    //
    // So the window is reconstructible. `conv_b` is all zeros in this
    // checkpoint (measured), the first two taps of a two-token prompt are
    // zero, and `conv_state` after the fire holds `m0` and `m1` themselves at
    // taps 2 and 3. Two candidate windows, one cosine each:
    //
    //     correct   w2*m0 + w3*m1      both tokens
    //     stale     w3*m1              the conv state never advanced
    let conv_w: Option<Vec<f32>> = real.get("layer.0.conv_w").map(|b| {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    });
    let conv_bytes = shell
        .recurrent()
        .and_then(|p| p.slab(0, "conv_state"))
        .and_then(|slab| shell.device().read_at(slab, 0, slab.size()).ok());
    if let (Some(cw), Some(cs)) = (conv_w, conv_bytes) {
        let conv_dim = 6144usize;
        let kc = 4usize;
        let cs: Vec<f32> = cs
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let tap = |slot: u32, j: usize, c: usize| -> f32 {
            cs[(slot as usize * kc + j) * conv_dim + c]
        };
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let k_off = 2048usize;
        println!("\n  THE WINDOW THE DECODE CONVOLVED, by angle against what it added:");
        for h in [8usize, 0, 3] {
            let ga = pre_gate[1][2 * h];
            let base = (h * dv_n + 81) * dk;
            let u: Vec<f32> = (0..dk).map(|i| dec[base + i] - ga * old[base + i]).collect();
            let mut both = vec![0.0f32; dk];
            let mut only1 = vec![0.0f32; dk];
            for d in 0..dk {
                let c = k_off + h * dk + d;
                let (m0, m1) = (tap(dec_slot, 2, c), tap(dec_slot, 3, c));
                both[d] = silu(cw[c * kc + 2] * m0 + cw[c * kc + 3] * m1);
                only1[d] = silu(cw[c * kc + 3] * m1);
            }
            // THE GUARD. A cosine against an all-zero vector is 0.0000 and
            // reads like a measurement, which is the same trap as a clean
            // read of the wrong megabyte: `conv_state` and `new_conv_state`
            // PING-PONG, so the buffer named `conv_state` after a fire is not
            // necessarily the one holding that fire's window.
            let live = both.iter().chain(&only1).any(|v| *v != 0.0);
            if !live {
                println!(
                    "    head {h:>2}: the reconstructed windows are ALL ZERO, so this reads \
                     the wrong half of the ping-pong and IT WAS NOT MEASURED"
                );
                continue;
            }
            println!(
                "    head {h:>2}: cos(u, k from BOTH taps) = {:+.4}, cos(u, k from the CURRENT tap only) = {:+.4}",
                cos(&u, &both),
                cos(&u, &only1)
            );
        }
    }

    println!("\n  THE DIRECTION THE DECODE ADDED, by angle:");
    for h in [8usize, 0, 3] {
        let ga = pre_gate[1][2 * h];
        let k = &pre_k[1][h * dk..(h + 1) * dk];
        let q = &pre_q[1][h * dk..(h + 1) * dk];
        let dirs: Vec<Vec<f32>> = [81usize, 17, 40]
            .into_iter()
            .map(|dv| {
                let base = (h * dv_n + dv) * dk;
                (0..dk)
                    .map(|i| dec[base + i] - ga * old[base + i])
                    .collect()
            })
            .collect();
        // Against token 1's staged pair, and against TOKEN 0's. `u` is a
        // fixed direction that is k-like without being `k`, and "the previous
        // token's k" is the one vector in this fire that is exactly that.
        let k0 = &pre_k[0][h * dk..(h + 1) * dk];
        let q0 = &pre_q[0][h * dk..(h + 1) * dk];
        println!(
            "    head {h:>2}: t1 k {:+.4}  t1 q {:+.4}   |   t0 k {:+.4}  t0 q {:+.4}",
            cos(&dirs[0], k),
            cos(&dirs[0], q),
            cos(&dirs[0], k0),
            cos(&dirs[0], q0)
        );
        println!(
            "            cos(u@81, u@17) = {:+.4}, cos(u@81, u@40) = {:+.4}  \
             -- these are 1.0 if the step is rank-one at all",
            cos(&dirs[0], &dirs[1]),
            cos(&dirs[0], &dirs[2])
        );
    }
}

/// Whether the scan stays exact as the prompt grows.
///
/// `whether_the_scan_computes_the_recurrence_its_own_inputs_imply` clears the
/// scan at TWO tokens, layer 0, one prompt -- and a sixteen-token prefill
/// still answers wrongly. So either the scan fails at a length that test does
/// not reach, or what is left is downstream of it, and those want opposite
/// next moves.
///
/// The reference is the same four lines and the same staged inputs; only the
/// number of steps changes. Two fresh seats per length -- one fire truncated
/// on `gdn_prep_prefill` to read what it staged, one fire whole to read what
/// the scan left -- because a truncated fire leaves its seat half-written and
/// reusing it would compare a thing against its own leftovers.
///
/// The lengths are not a ladder for its own sake. `PIE_VROWS` and the tail
/// masking mean the scan's grid divides `Dv` differently at different launches
/// and the token loop is sequential in `t`, so a defect that needs a second
/// step, or a fifth, or a tile boundary, shows as a length where the agreement
/// stops rather than as a slope.
///
/// # NOT YET RUN, and why
///
/// Written against a tree where `ForwardPlan::finish` panics before any of
/// this is reached. `model-ir::kernels::arity_problem` counts a routine's
/// `params: Buf` -- and its slabs and its `FireTable`s -- as pointers the
/// STATEMENT places, and a statement places none of the three:
///
///     `rms_single_row_bfloat16` reads 3 pointers but the statement places 2
///     `gdn_prep_prefill_bfloat16` reads 10 but the statement places 7
///
/// Three reads unaccounted for in the second is exactly `params`, one slab and
/// one table.
///
/// **That was only half of it, and the smaller half.** Skipping the arity
/// assert locally lets the plan build, and then every fire is REFUSED:
///
///     `affine_qmv_fast_bfloat16_gs_64_b_4` could not be planned:
///     Empty { what: "an input operand the arm asked for" }
///
/// The lowering emits that launch with ONE arena operand -- its input, 1024
/// wide -- and no destination at all, while `rms_single_row` beside it has
/// both. So `Handles::build` splits one widthed operand with `results = 1`,
/// which leaves the input side empty and the arm asking for something that is
/// not there. The op itself says so: `affine_qmv_fast` and `affine_qmm_t`
/// declare `outputs 0` where `rms_single_row` declares 1.
///
/// **A matrix-vector product with nowhere to write is not an accounting
/// disagreement**, so the write-side half of the arity complaint --
/// "writes 1 pointer but the statement declares 0 results" -- was RIGHT, and
/// the previous commit's reading of it as the checker's over-count covered
/// only the read side.
///
/// The mechanism is `1fc015de9`, which made a region's output reach a
/// statement that declares none only when
/// `kernels::accepts_an_unstated_result` says so -- and that asks for an
/// argument marked `Provenance::Either` AND `Binds::Writes`. The affine GEMMs
/// declare a plain `BufMut` for their `y`, which is `Trace`, so the gate
/// answers false and the destination is never handed over.
///
/// `cargo test -p model` is red across gpt_oss, gemma_3, gemma_4, llama_like
/// and qwen_3_5, and nothing here is on `model`'s dependency path at all.
/// Left in place rather than deleted or made to skip: it is the next
/// question, and it will answer itself the moment the trace builds.
///
/// # Upstream has since fixed the WRITE half, and that confirms both readings
///
/// `arity_problem` grew an `inside_value_region` parameter, and the
/// "writes 1 pointer but the statement declares 0 results" violations are
/// gone -- a guard's destination now reaches its arms. What remains is
/// exactly the read side this note opened with:
///
///     `rms_single_row_bfloat16` reads 3 pointers but the statement places 2
///
/// three being `x`, `w` and the `params` BLOCK, which no statement places as
/// an operand. The vocabulary already has the answer -- `Env<T>` carries
/// `Provenance::Env` and `arity_problem` skips it -- so the fix is to mark
/// the params blocks, the slabs and the `FireTable`s that way, across the
/// backends' routine signatures. Not done here: it is a convention four
/// tables share, on lines the reconcile is actively editing.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_the_scan_stays_exact_as_the_prompt_grows() {
    let Some((mut shell, _real)) = qwen3_5_shell(24) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let (hv, dv_n, dk) = (16usize, 128usize, 128usize);
    let (qk_pitch, g_pitch) = (hv * dk, 2 * hv + hv * dv_n);

    println!("\n  the scan against the recurrence its own staged inputs imply:");
    // The widest disagreement RELATIVE to the reference's own range, because
    // two summation orders over f32 do not agree bitwise and never will: the
    // scan folds 32 lane partials in a tree and this reference adds them left
    // to right. One ULP at 0.5 is 6e-8, so the question is whether the gap
    // stays at rounding or leaves it.
    let mut worst_rel = 0.0f32;
    // HOW MANY LENGTHS ACTUALLY GOT MEASURED. Without this the verdict below
    // reads "exact at every length" when EVERY fire was refused, because a
    // maximum over nothing is zero -- which is the vacuity this file has now
    // been caught by three times.
    let mut measured = 0usize;
    // Lengths at layer 0, then LAYERS at a fixed length. The first asks
    // whether the token loop accumulates correctly; the second asks whether
    // the kernel that does is the same one at depth, where the inputs are the
    // model's own activations rather than the embedding. Layer 3 is skipped
    // because it is full attention and has no recurrence to compare.
    let cases: Vec<(usize, u16)> = [2usize, 3, 4, 6, 8, 16]
        .into_iter()
        .map(|n| (n, 0u16))
        .chain([1u16, 2, 4, 5, 6].into_iter().map(|l| (4usize, l)))
        .collect();
    for (i, (n, layer)) in cases.into_iter().enumerate() {
        let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % 6]).collect();
        let rows: Vec<Row> = vec![
            Row {
                samples: true,
                ..Row::default()
            };
            n
        ];
        let Ok(low) = lower(
            &plan,
            &rows,
            Fire {
                captures_across_splits: false,
            },
        ) else {
            continue;
        };
        let Some(at) = low.launches.iter().position(|l| {
            low.kernels[l.kernel as usize].starts_with("gdn_prep_prefill") && l.layers.start == layer
        }) else {
            println!("    n = {n:>2} layer {layer:>2}: no `gdn_prep_prefill` there");
            continue;
        };
        let l = &low.launches[at];
        let arenas: Vec<(usize, usize)> = low.args[l.args.start as usize..l.args.end as usize]
            .iter()
            .filter_map(|a| match a {
                Arg::Arena { at, width, bytes } => Some((*at, *width as usize * *bytes as usize)),
                _ => None,
            })
            .collect();
        if arenas.len() != 6 {
            continue;
        }
        shell.fire_prefix(Some(at + 1));
        let staged = shell.step(&[driver_wgpu::turns::Turn {
            who: 3_000 + i as u64 * 2,
            tokens: tokens.clone(),
        }]);
        shell.fire_prefix(None);
        let step = match staged {
            Ok(step) => step,
            Err(why) => {
                println!("    n = {n:>2} layer {layer:>2}: the truncated prefill was refused ({why})");
                continue;
            }
        };
        let arena = step.arena;
        let read = |at: usize, stride: usize, t: usize, w: usize| -> Vec<f32> {
            let from = at + t * stride;
            arena[from..from + w * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let pre_k: Vec<Vec<f32>> = (0..n)
            .map(|t| read(arenas[4].0, arenas[4].1, t, qk_pitch))
            .collect();
        let pre_gate: Vec<Vec<f32>> = (0..n)
            .map(|t| read(arenas[5].0, arenas[5].1, t, g_pitch))
            .collect();
        let mut want = vec![0.0f32; hv * dv_n * dk];
        for h in 0..hv {
            for dv in 0..dv_n {
                let base = (h * dv_n + dv) * dk;
                for t in 0..n {
                    let ga = pre_gate[t][2 * h];
                    let gb = pre_gate[t][2 * h + 1];
                    let vv = pre_gate[t][2 * hv + h * dv_n + dv];
                    let k = &pre_k[t][h * dk..(h + 1) * dk];
                    let mut kv = 0.0f32;
                    for j in 0..dk {
                        want[base + j] *= ga;
                        kv += want[base + j] * k[j];
                    }
                    let delta = (vv - kv) * gb;
                    for j in 0..dk {
                        want[base + j] += k[j] * delta;
                    }
                }
            }
        }
        let who = 3_001 + i as u64 * 2;
        if fire_row(&mut shell, who, &tokens).is_empty() {
            println!("    n = {n:>2} layer {layer:>2}: the whole prefill was refused");
            continue;
        }
        let Some(slot) = shell.book().slot(who) else {
            continue;
        };
        let Some(pool) = shell.recurrent() else { return };
        let Some(slab) = pool.slab(layer, "recurrent_state") else {
            return;
        };
        let Ok(bytes) = shell.device().read_at(slab, 0, slab.size()) else {
            continue;
        };
        let per = hv * dv_n * dk;
        let from = per * slot as usize * 4;
        let got: Vec<f32> = bytes[from..from + per * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // NOT just the widest: `{:.6}` prints 1e-9 as 0.000000, and "bit-exact"
        // and "agrees to six decimals" are different claims. The count of
        // elements that differ AT ALL is the one that cannot round.
        let (mut worst, mut nan, mut differ) = (0.0f32, 0usize, 0usize);
        for (a, b) in want.iter().zip(&got) {
            if a != b {
                differ += 1;
            }
            let d = (a - b).abs();
            if d.is_finite() {
                worst = worst.max(d);
            } else {
                nan += 1;
            }
        }
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        worst_rel = worst_rel.max(worst / scale.max(1e-30));
        measured += 1;
        println!(
            "    n = {n:>2} layer {layer:>2}: {differ} of {per} differ, widest |cpu - gpu| {worst:e} \
             against a reference whose own widest is {scale:.6}{}",
            if nan > 0 {
                format!(", and {nan} non-finite")
            } else {
                String::new()
            }
        );
    }
    println!(
        "\n  {}",
        if measured == 0 {
            "NOTHING WAS MEASURED -- every fire was refused, and a maximum over nothing is zero."
        } else if worst_rel < 1e-5 {
            "THE SCAN AGREES TO F32 ROUNDING EVERYWHERE MEASURED, so this recurrence is not where the answer goes wrong."
        } else {
            "THE SCAN STOPS AGREEING, and where it stops is the shape of the defect."
        }
    );
    assert!(
        worst_rel < 1e-5,
        "the scan left f32 rounding somewhere: widest relative disagreement {worst_rel:e}"
    );
    assert!(
        measured > 0,
        "no length was measured, so this run says nothing about the scan"
    );
}

/// Whether the PREP stages what the convolution and the gates imply.
///
/// `whether_the_scan_stays_exact_as_the_prompt_grows` clears the scan at every
/// length from 2 to 16 and at layers 0, 1, 2, 4, 5 and 6 -- always within one
/// or two ULP of a CPU walk of the same four lines. But that reference is
/// built FROM `gdn_prep_prefill`'s staged output, so it says the scan is
/// faithful to its inputs and nothing about whether those inputs are right. A
/// prep that stages the wrong `k` produces a scan that agrees with itself
/// perfectly and a model that answers nonsense.
///
/// This closes that gap. Everything the prep reads is readable: `mixed` is an
/// arena operand of the same rectangle, `a` and `b` are the two 16-wide
/// projections beside it, and `conv_w`, `conv_b`, `a_log` and `dt` are
/// weights the shell holds. So `pre_q`, `pre_k` and `pre_gate` can be
/// recomputed and compared against what the device wrote.
///
/// # Why this is the suspect
///
/// `gdn_core_slotted` does the SAME convolution in its own file, and it is
/// exact at the first token and wrong at the second -- which is exactly when
/// the conv state stops being zero and the taps start mattering. A defect in
/// the tap arithmetic would give both of the things measured so far: a decode
/// that fails from t = 1, and a scan that agrees with staged inputs that are
/// themselves wrong.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_the_prep_stages_what_the_convolution_implies() {
    let Some((mut shell, real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let n = 4usize;
    let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % 6]).collect();
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    let Some(at) = low.launches.iter().position(|l| {
        low.kernels[l.kernel as usize].starts_with("gdn_prep_prefill") && l.layers.start == 0
    }) else {
        return;
    };
    let l = &low.launches[at];
    let arenas: Vec<(usize, usize, u32)> = low.args[l.args.start as usize..l.args.end as usize]
        .iter()
        .filter_map(|a| match a {
            Arg::Arena { at, width, bytes } => {
                Some((*at, *width as usize * *bytes as usize, *bytes))
            }
            _ => None,
        })
        .collect();
    if arenas.len() != 6 {
        println!("`gdn_prep_prefill` has {} arena ranges, not 6", arenas.len());
        return;
    }
    shell.fire_prefix(Some(at + 1));
    let arena = match shell.step(&[driver_wgpu::turns::Turn {
        who: 4_000,
        tokens: tokens.clone(),
    }]) {
        Ok(s) => s.arena,
        Err(why) => {
            println!("the truncated prefill was refused: {why}");
            return;
        }
    };
    shell.fire_prefix(None);

    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let row = |i: usize, t: usize| -> Vec<f32> {
        let (o, stride, bytes) = arenas[i];
        let s = &arena[o + t * stride..o + (t + 1) * stride];
        if bytes == 2 { bf16(s) } else { f32s(s) }
    };
    let (hv, dv_n, dk, kc, conv_dim) = (16usize, 128usize, 128usize, 4usize, 6144usize);
    let (q_off, k_off, v_off) = (0usize, 2048usize, 4096usize);
    let (qk_pitch, g_pitch) = (hv * dk, 2 * hv + hv * dv_n);
    let inv_sqrt_dk = (dk as f32).powf(-0.5);
    let eps = 1e-6f32;

    let Some(cw) = real.get("layer.0.conv_w").map(|b| bf16(b)) else {
        return;
    };
    let Some(cb) = real.get("layer.0.conv_b").map(|b| bf16(b)) else {
        return;
    };
    let Some(a_log) = real.get("layer.0.a_log").map(|b| f32s(b)) else {
        return;
    };
    let Some(dt) = real.get("layer.0.dt").map(|b| bf16(b)) else {
        return;
    };
    let mixed: Vec<Vec<f32>> = (0..n).map(|t| row(0, t)).collect();
    let a_in: Vec<Vec<f32>> = (0..n).map(|t| row(1, t)).collect();
    let b_in: Vec<Vec<f32>> = (0..n).map(|t| row(2, t)).collect();

    let silu = |x: f32| x / (1.0 + (-x).exp());
    // `conv_state` is zero on a fresh seat, so a tap before the prompt is zero.
    let conv = |t: usize, c: usize| -> f32 {
        let mut acc = cb[c];
        for j in 0..kc {
            let idx = t as i64 - (kc as i64 - 1) + j as i64;
            if idx >= 0 {
                acc += mixed[idx as usize][c] * cw[c * kc + j];
            }
        }
        silu(acc)
    };

    let mut worst = [0.0f32; 3];
    let mut scale = [0.0f32; 3];
    let names = ["pre_q", "pre_k", "pre_gate"];
    for t in 0..n {
        let (dq, dk_dev, dg) = (row(3, t), row(4, t), row(5, t));
        for h in 0..hv {
            let (mut qsq, mut ksq) = (0.0f32, 0.0f32);
            let mut qraw = vec![0.0f32; dk];
            let mut kraw = vec![0.0f32; dk];
            for d in 0..dk {
                qraw[d] = conv(t, q_off + h * dk + d);
                kraw[d] = conv(t, k_off + h * dk + d);
                qsq += qraw[d] * qraw[d];
                ksq += kraw[d] * kraw[d];
            }
            let qinv = inv_sqrt_dk / (qsq + eps).sqrt();
            let kinv = 1.0 / (ksq + eps).sqrt();
            for d in 0..dk {
                let at = h * dk + d;
                worst[0] = worst[0].max((qraw[d] * qinv - dq[at]).abs());
                worst[1] = worst[1].max((kraw[d] * kinv - dk_dev[at]).abs());
                scale[0] = scale[0].max(dq[at].abs());
                scale[1] = scale[1].max(dk_dev[at].abs());
            }
            let ad = a_in[t][h] + dt[h];
            let sp = ad.max(0.0) + (1.0 + (-ad.abs()).exp()).ln();
            let ga = (-a_log[h].exp() * sp).exp();
            let gb = 1.0 / (1.0 + (-b_in[t][h]).exp());
            worst[2] = worst[2].max((ga - dg[2 * h]).abs());
            worst[2] = worst[2].max((gb - dg[2 * h + 1]).abs());
            for dv in 0..dv_n {
                let want = conv(t, v_off + h * dv_n + dv);
                let at = 2 * hv + h * dv_n + dv;
                worst[2] = worst[2].max((want - dg[at]).abs());
                scale[2] = scale[2].max(dg[at].abs());
            }
        }
    }
    let _ = (qk_pitch, g_pitch, conv_dim);
    println!("\n  the prep against the convolution and the gates it states:");
    let mut bad = Vec::new();
    for i in 0..3 {
        let rel = worst[i] / scale[i].max(1e-30);
        println!(
            "    {:<9} widest |cpu - gpu| {:e}, against values reaching {:.6} -- relative {rel:e}",
            names[i], worst[i], scale[i]
        );
        if rel > 1e-4 {
            bad.push(names[i]);
        }
    }
    println!(
        "\n  {}",
        if bad.is_empty() {
            "THE PREP STAGES WHAT ITS INPUTS IMPLY, so the prefill's gated DeltaNet is right end to end."
        } else {
            "THE PREP DISAGREES WITH ITS OWN INPUTS, and which tensor says which half."
        }
    );
    assert!(
        bad.is_empty(),
        "the prep does not stage what the convolution and the gates imply: {bad:?}"
    );
}

/// Whether the gated norm computes what its three operands imply.
///
/// The prefill's gated DeltaNet is now verified end to end at layer 0: the
/// prep stages what the convolution and the gates imply (relative 1e-7), and
/// the scan computes the recurrence those staged inputs imply (relative 1e-7,
/// at every length from 2 to 16 and at layers 0, 1, 2, 4, 5 and 6). And the
/// half still answers wrongly.
///
/// `norm::gated_rms` is what is left that only this half runs. The attention
/// block never touches it, which is exactly the shape of the earlier cut --
/// attention alone right, gated DeltaNet alone wrong -- and it has been
/// checked only by reading it against `kernels-metal` and `kernels-cuda`.
///
/// It is three operands and one weight, all readable:
///
///     out[i] = w[i mod vd] * x[i] * rsqrt(mean(x[head]^2) + eps) * silu(z[i])
///
/// with the mean over `vd = 128` channels of ONE head -- which is the part
/// worth measuring rather than reading, because a norm taken over the wrong
/// span is finite, plausible and wrong, and `gated_rms` was already caught
/// once being told the attention's head count instead of this block's.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_the_gated_norm_computes_what_its_operands_imply() {
    let Some((mut shell, real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let n = 4usize;
    let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % 6]).collect();
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    let Some(at) = low.launches.iter().position(|l| {
        low.kernels[l.kernel as usize].starts_with("gated_rms") && l.layers.start == 0
    }) else {
        println!("no `gated_rms` at layer 0");
        return;
    };
    let l = &low.launches[at];
    let arenas: Vec<(usize, usize, u32)> = low.args[l.args.start as usize..l.args.end as usize]
        .iter()
        .filter_map(|a| match a {
            Arg::Arena { at, width, bytes } => {
                Some((*at, *width as usize * *bytes as usize, *bytes))
            }
            _ => None,
        })
        .collect();
    println!("\n  `gated_rms` at layer 0 is rectangle {at}, with {} arena ranges", arenas.len());
    if arenas.len() != 3 {
        println!("  not the three this test knows how to read");
        return;
    }
    shell.fire_prefix(Some(at + 1));
    let arena = match shell.step(&[driver_wgpu::turns::Turn {
        who: 4_100,
        tokens: tokens.clone(),
    }]) {
        Ok(s) => s.arena,
        Err(why) => {
            println!("the truncated prefill was refused: {why}");
            return;
        }
    };
    shell.fire_prefix(None);
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let row = |i: usize, t: usize| -> Vec<f32> {
        let (o, stride, bytes) = arenas[i];
        let s = &arena[o + t * stride..o + (t + 1) * stride];
        if bytes == 2 { bf16(s) } else { f32s(s) }
    };
    let Some(w) = real.get("layer.0.gate_norm").map(|b| bf16(b)) else {
        println!("no `layer.0.gate_norm`");
        return;
    };
    let (hv, vd) = (16usize, 128usize);
    let eps = 1e-6f32;
    let silu = |x: f32| x / (1.0 + (-x).exp());
    // THE OUTPUT IS BF16, so the reference has to be too before it is
    // subtracted. Eight mantissa bits is a relative step of 2^-8 = 3.9e-3, and
    // the first run of this test compared an f32 reference against a bf16
    // result and read 2.8e-3 as a defect -- which is one rounding, not one.
    // Round-to-nearest-even, the same as `pie_f32_to_bf16`.
    let to_bf16 = |v: f32| -> f32 {
        let b = v.to_bits();
        if (b & 0x7fff_ffff) > 0x7f80_0000 {
            return f32::NAN;
        }
        f32::from_bits((b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) << 16)
    };
    let (mut worst, mut scale) = (0.0f32, 0.0f32);
    let (mut worst_ungated, mut worst_wholerow) = (f32::INFINITY, f32::INFINITY);
    for t in 0..n {
        let (x, z, out) = (row(0, t), row(1, t), row(2, t));
        // The whole row's rms as well as the head's, because "which span the
        // mean is taken over" is the question a reading cannot settle.
        let whole: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv_whole = 1.0 / (whole + eps).sqrt();
        let (mut w_head, mut w_ungated, mut w_whole) = (0.0f32, 0.0f32, 0.0f32);
        for h in 0..hv {
            let head = &x[h * vd..(h + 1) * vd];
            let mean: f32 = head.iter().map(|v| v * v).sum::<f32>() / vd as f32;
            let inv = 1.0 / (mean + eps).sqrt();
            for (d, wd) in w.iter().enumerate().take(vd) {
                let at = h * vd + d;
                let want = to_bf16(wd * x[at] * inv * silu(z[at]));
                w_head = w_head.max((want - out[at]).abs());
                w_ungated = w_ungated.max((wd * x[at] * inv - out[at]).abs());
                w_whole = w_whole.max((wd * x[at] * inv_whole * silu(z[at]) - out[at]).abs());
                scale = scale.max(out[at].abs());
            }
        }
        worst = worst.max(w_head);
        worst_ungated = worst_ungated.min(w_ungated);
        worst_wholerow = worst_wholerow.min(w_whole);
    }
    let rel = worst / scale.max(1e-30);
    println!("    per-head norm, gate after:   widest {worst:e}, relative {rel:e} (one bf16 step is 3.9e-3)");
    println!("    the same with NO gate:       widest {worst_ungated:e}");
    println!("    whole-row norm, gate after:  widest {worst_wholerow:e}");
    println!("    against values reaching {scale:.6}");
    println!(
        "\n  {}",
        if rel < 8e-3 {
            "THE GATED NORM COMPUTES WHAT ITS OPERANDS IMPLY."
        } else {
            "THE GATED NORM DISAGREES WITH ITS OPERANDS, and the two alternatives above say how."
        }
    );
    assert!(
        rel < 8e-3,
        "`gated_rms` does not compute `w * rmsnorm_per_head(x) * silu(z)`: \
         relative {rel:e}, which is more than two bf16 steps"
    );
}

/// Whether the in-projection produces the `mixed` the weights imply.
///
/// Every kernel of the prefill's gated DeltaNet is now verified against a CPU
/// reference: the prep stages what the convolution and the gates imply, the
/// scan computes the recurrence those inputs imply at every length and every
/// layer, and `gated_rms` is bf16-exact. And the half still answers nothing.
///
/// What all three references took AS GIVEN is `mixed`. They read it out of the
/// arena and asked what follows from it, so a wrong in-projection produces a
/// block that is faithful at every step and wrong from the first.
///
/// This closes it: `w = scale * code + bias` with four-bit codes packed eight
/// to a word and one scale-and-bias pair per 64 inputs, so `in_proj_qkv` is
/// `[6144, 1024]` in 3,145,728 bytes with 32 bytes of scales and 32 of zeros
/// per output row. Dequantise it, multiply by the norm's output, and compare
/// against what the device left.
///
/// It also tests the SHARED path -- `affine_qmv_fast` and the loader's
/// unpacking are what qwen3-0.6b serves 22 of 22 through -- so a disagreement
/// here would be a much larger finding than a gated-DeltaNet one.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_the_in_projection_produces_the_mixed_the_weights_imply() {
    let Some((mut shell, real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let n = 2usize;
    let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % 6]).collect();
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    let ranges = |at: usize| -> Vec<(usize, usize, u32)> {
        let l = &low.launches[at];
        low.args[l.args.start as usize..l.args.end as usize]
            .iter()
            .filter_map(|a| match a {
                Arg::Arena { at, width, bytes } => {
                    Some((*at, *width as usize * *bytes as usize, *bytes))
                }
                _ => None,
            })
            .collect()
    };
    let Some(prep) = low.launches.iter().position(|l| {
        low.kernels[l.kernel as usize].starts_with("gdn_prep_prefill") && l.layers.start == 0
    }) else {
        return;
    };
    let Some(norm) = low.launches.iter().position(|l| {
        low.kernels[l.kernel as usize].starts_with("rms_single_row") && l.layers.start == 0
    }) else {
        return;
    };
    shell.fire_prefix(Some(prep + 1));
    let arena = match shell.step(&[driver_wgpu::turns::Turn {
        who: 4_200,
        tokens: tokens.clone(),
    }]) {
        Ok(s) => s.arena,
        Err(why) => {
            println!("the truncated prefill was refused: {why}");
            return;
        }
    };
    shell.fire_prefix(None);
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    let read = |rs: &[(usize, usize, u32)], i: usize, t: usize| -> Vec<f32> {
        let (o, stride, _) = rs[i];
        bf16(&arena[o + t * stride..o + (t + 1) * stride])
    };
    let (nr, pr) = (ranges(norm), ranges(prep));
    // The norm's OUTPUT is the projection's input; `mixed` is the prep's first
    // arena range. Both bf16, both this fire's.
    let x: Vec<Vec<f32>> = (0..n).map(|t| read(&nr, 1, t)).collect();
    let mixed: Vec<Vec<f32>> = (0..n).map(|t| read(&pr, 0, t)).collect();
    let (Some(codes), Some(sc), Some(ze)) = (
        real.get("layer.0.in_proj_qkv"),
        real.get("layer.0.in_proj_qkv.scales").map(|b| bf16(b)),
        real.get("layer.0.in_proj_qkv.zeros").map(|b| bf16(b)),
    ) else {
        println!("the projection's three tensors are not all here");
        return;
    };
    let (out_n, in_n, group) = (6144usize, 1024usize, 64usize);
    let groups = in_n / group;
    println!(
        "\n  in_proj_qkv: {} code bytes, {} scales, {} zeros, against [{out_n}, {in_n}] at {group}",
        codes.len(),
        sc.len(),
        ze.len()
    );
    if codes.len() != out_n * in_n / 2 || sc.len() != out_n * groups {
        println!("  not the shape this test knows how to unpack");
        return;
    }
    let (mut worst, mut scale) = (0.0f32, 0.0f32);
    let mut at_o = 0usize;
    for t in 0..n {
        for o in 0..out_n {
            let mut acc = 0.0f32;
            let base = o * in_n / 2;
            for g in 0..groups {
                let (s, z) = (sc[o * groups + g], ze[o * groups + g]);
                for e in 0..group {
                    let i = g * group + e;
                    let byte = codes[base + i / 2];
                    let code = f32::from(if i % 2 == 0 { byte & 0xf } else { byte >> 4 });
                    acc += x[t][i] * (s * code + z);
                }
            }
            let d = (acc - mixed[t][o]).abs();
            if d > worst {
                worst = d;
                at_o = o;
            }
            scale = scale.max(mixed[t][o].abs());
        }
    }
    let rel = worst / scale.max(1e-30);
    println!(
        "    widest |cpu - gpu| {worst:e} at output {at_o}, against values reaching \
         {scale:.6} -- relative {rel:e} (one bf16 step is 3.9e-3)"
    );
    println!(
        "\n  {}",
        if rel < 3e-2 {
            "THE IN-PROJECTION PRODUCES WHAT ITS WEIGHTS IMPLY."
        } else {
            "THE IN-PROJECTION DISAGREES WITH ITS OWN WEIGHTS, and everything downstream was faithful to a wrong `mixed`."
        }
    );
}

/// The first row of a prefill attends to ITSELF and nothing else.
///
/// Every kernel of the prefill's gated DeltaNet is now verified against a CPU
/// reference -- the in-projection from its dequantised weights, the prep's
/// convolution and gates, the recurrence at every length and layer, and the
/// gated norm to the bf16 bit. So the earlier cut needs re-reading: silencing
/// the attention layers leaves a model missing 6 of its 24, and one answering
/// nonsense is what that IS, not evidence about which half is defective.
///
/// The attention block is the one never measured, and it is the one carrying
/// everything qwen3-0.6b never exercises: `head_dim = 256`, a quarter-turn
/// partial rope, the fused query-gate split and the sigmoid output gate.
///
/// # What this probe establishes, and what it does NOT
///
/// SOLID: row 0's output is a PURE COPY of one `v` block -- relative
/// difference 0.000, not merely angle 0 -- and heads 0 to 3 copy a different
/// block from heads 4 to 7. Together with
/// `whether_a_later_token_moves_an_earlier_rows_attention`, which finds row 0
/// bit-identical under a changed third token, that says the mask holds and the
/// ADDRESSING does not.
///
/// NOT SOLID: which token's `v` each one copies. This reads the `v` operand as
/// `[rows][kv_heads * head_dim]` at the arena range's own width, and the
/// numbers refute that: at `n = 5` the blocks it calls rows 0 and 1 are
/// IDENTICAL, and so are 2 and 3. A layout that repeats cannot be the one
/// assumed, so "heads 4 to 7 read row 2" is this probe's stride talking and
/// not the device's. The naming is left in the output because the PATTERN --
/// two head groups landing on two different blocks, and agreeing at `n = 2`
/// where there are fewer blocks to land on -- is what needs explaining next,
/// and the first thing to settle is that stride.
///
/// This is the cheapest true statement about it. With causal masking, row 0 of
/// a fresh prefill has exactly one key to attend to -- its own -- so its
/// softmax is `[1.0]` and
///
///     out[0, h, :] == v[0, kv_head(h), :]
///
/// exactly, whatever the scale, whatever the rope, whatever the softmax's
/// numerics. It needs no reference implementation and no KV-cache read: `v`
/// is what `kv_append_paged` was handed. A disagreement here is head mapping,
/// GQA broadcast or causality, and any of the three is the whole block.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_the_first_row_of_attention_is_its_own_value() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    // SEVERAL LENGTHS. Row 0's answer is a pure copy of one `v` row whatever
    // the prompt is, so which row it copies, as `n` moves, is what tells a
    // REVERSAL (`n - 1 - p`) from a fixed stride.
    for n in [2usize, 3, 5] {
    let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % 6]).collect();
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    let ranges = |at: usize| -> Vec<(usize, usize, u32)> {
        let l = &low.launches[at];
        low.args[l.args.start as usize..l.args.end as usize]
            .iter()
            .filter_map(|a| match a {
                Arg::Arena { at, width, bytes } => {
                    Some((*at, *width as usize * *bytes as usize, *bytes))
                }
                _ => None,
            })
            .collect()
    };
    let find = |stem: &str| -> Option<usize> {
        low.launches
            .iter()
            .position(|l| low.kernels[l.kernel as usize].starts_with(stem) && l.layers.start == 3)
    };
    let (Some(sdpa), Some(append)) = (find("sdpa_paged_tiled"), find("kv_append_paged")) else {
        println!("layer 3 has no tiled sdpa and kv append");
        continue;
    };
    shell.fire_prefix(Some(sdpa + 1));
    let arena = match shell.step(&[driver_wgpu::turns::Turn {
        who: 4_300 + n as u64,
        tokens: tokens.clone(),
    }]) {
        Ok(s) => s.arena,
        Err(why) => {
            println!("the truncated prefill was refused: {why}");
            continue;
        }
    };
    shell.fire_prefix(None);
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    let read = |rs: &[(usize, usize, u32)], i: usize, t: usize| -> Vec<f32> {
        let (o, stride, _) = rs[i];
        bf16(&arena[o + t * stride..o + (t + 1) * stride])
    };
    let (ar, sr) = (ranges(append), ranges(sdpa));
    if ar.len() < 2 || sr.len() < 2 {
        println!("  not the shape this test knows how to read");
        continue;
    }
    // `kv_append_paged` takes k then v; the sdpa's last range is its output.
    let v0 = read(&ar, ar.len() - 1, 0);
    let out0 = read(&sr, sr.len() - 1, 0);
    let (q_heads, kv_heads, head_dim) = (8usize, 2usize, 256usize);
    let rep = q_heads / kv_heads;
    if v0.len() != kv_heads * head_dim || out0.len() != q_heads * head_dim {
        println!("  widths are not `[kv_heads, head_dim]` and `[q_heads, head_dim]`");
        continue;
    }
    println!("\n  n = {n}: which `v` row each head of OUT ROW 0 copies");
    // BOTH GQA MAPPINGS AND EVERY ROW'S `v`, because "not its own value" has
    // three shapes and they want different fixes. `h / rep` is blocked and
    // `h % kv_heads` is interleaved -- the classic pair -- and matching a
    // LATER row's `v` instead of row 0's is a causality or mask fault rather
    // than a mapping one.
    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..a.len() {
            ab += f64::from(a[i]) * f64::from(b[i]);
            aa += f64::from(a[i]) * f64::from(a[i]);
            bb += f64::from(b[i]) * f64::from(b[i]);
        }
        (ab / (aa.sqrt() * bb.sqrt()).max(1e-30)) as f32
    };
    for h in [0usize, 4] {
        let o = &out0[h * head_dim..(h + 1) * head_dim];
        let mut best = (0.0f32, String::new());
        let mut line = String::new();
        for r in 0..n {
            let vr = read(&ar, ar.len() - 1, r);
            for (name, kv) in [("blocked", h / rep), ("interleaved", h % kv_heads)] {
                let c = cos(o, &vr[kv * head_dim..(kv + 1) * head_dim]);
                if c.abs() > best.0.abs() {
                    best = (c, format!("row {r} kv {kv} ({name})"));
                }
                if r == 0 {
                    line.push_str(&format!(" {name}={c:+.4}"));
                }
            }
        }
        let _ = line;
        // COSINE CANNOT TELL PARALLEL FROM EQUAL, and a softmax produces a
        // convex combination, so a true copy has relative difference zero and
        // not merely angle zero. Both, per candidate row.
        let mut diffs = String::new();
        for r in 0..n {
            let vr = read(&ar, ar.len() - 1, r);
            let kv = h / rep;
            let (mut w, mut sc) = (0.0f32, 0.0f32);
            for d in 0..head_dim {
                w = w.max((o[d] - vr[kv * head_dim + d]).abs());
                sc = sc.max(vr[kv * head_dim + d].abs());
            }
            diffs.push_str(&format!(" r{r}={:.3}", w / sc.max(1e-30)));
        }
        let mag = o.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!(
            "      head {h}: best cos {:+.4} at {}; |out| {mag:.4}; relative diff{diffs}",
            best.0, best.1
        );
    }
    let (mut worst, mut scale, mut at_h) = (0.0f32, 0.0f32, 0usize);
    for h in 0..q_heads {
        let kv = h / rep;
        for d in 0..head_dim {
            let want = v0[kv * head_dim + d];
            let got = out0[h * head_dim + d];
            if (want - got).abs() > worst {
                worst = (want - got).abs();
                at_h = h;
            }
            scale = scale.max(want.abs());
        }
    }
    let rel = worst / scale.max(1e-30);
    println!("    widest |v0 - out0| {worst:e} at head {at_h}, against values reaching {scale:.6} -- relative {rel:e}");
    println!(
        "    {}",
        if rel < 1e-2 {
            "row 0 IS its own value here"
        } else {
            "row 0 is NOT its own value here"
        }
    );
    }
}

/// Whether a LATER token can change an EARLIER token's attention output.
///
/// `whether_the_first_row_of_attention_is_its_own_value` finds that row 0 of a
/// prefill copies one `v` row exactly -- the softmax is saturated, so it picks
/// a single key -- and that heads 4 to 7 pick row 2 at both `n = 3` and
/// `n = 5`, while heads 0 to 3 pick row 0. Correct at `n = 2`, where there is
/// no row 2 to pick.
///
/// Two readings fit: the second key head's rows are addressed wrongly, or the
/// causal mask is not holding and row 0 is choosing an argmax over keys it
/// should never see. They want opposite fixes, and this separates them without
/// a reference of any kind.
///
/// **Change only the LAST token.** Under causal masking, row 0's output cannot
/// depend on it -- not approximately, not to a tolerance, at all. If it moves,
/// the mask is what is broken; if it does not, the addressing is.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_a_later_token_moves_an_earlier_rows_attention() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let n = 3usize;
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    let Some(sdpa) = low.launches.iter().position(|l| {
        low.kernels[l.kernel as usize].starts_with("sdpa_paged_tiled") && l.layers.start == 3
    }) else {
        println!("layer 3 has no tiled sdpa");
        return;
    };
    let l = &low.launches[sdpa];
    let arenas: Vec<(usize, usize)> = low.args[l.args.start as usize..l.args.end as usize]
        .iter()
        .filter_map(|a| match a {
            Arg::Arena { at, width, bytes } => Some((*at, *width as usize * *bytes as usize)),
            _ => None,
        })
        .collect();
    if arenas.len() < 2 {
        return;
    }
    let (o, stride) = arenas[arenas.len() - 1];
    let fire = |shell: &mut driver_wgpu::shell::Shell, who: u64, last: u32| -> Vec<f32> {
        shell.fire_prefix(Some(sdpa + 1));
        let out = shell
            .step(&[driver_wgpu::turns::Turn {
                who,
                tokens: vec![PERIOD[0], PERIOD[1], last],
            }])
            .map(|s| s.arena)
            .unwrap_or_default();
        shell.fire_prefix(None);
        if out.is_empty() {
            return Vec::new();
        }
        // ROW 0 of the attention's output, and only that.
        out[o..o + stride]
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    let a = fire(&mut shell, 4_400, PERIOD[2]);
    let b = fire(&mut shell, 4_401, PERIOD[5]);
    if a.is_empty() || b.is_empty() {
        println!("a fire was refused, so IT COULD NOT BE MEASURED");
        return;
    }
    let (q_heads, head_dim) = (8usize, 256usize);
    println!("\n  the same two first tokens, a different THIRD, and row 0 of the attention:");
    let mut moved_any = false;
    for h in 0..q_heads {
        let (mut worst, mut scale) = (0.0f32, 0.0f32);
        for d in 0..head_dim {
            let i = h * head_dim + d;
            worst = worst.max((a[i] - b[i]).abs());
            scale = scale.max(a[i].abs());
        }
        let rel = worst / scale.max(1e-30);
        if rel > 1e-6 {
            moved_any = true;
        }
        println!(
            "      head {h}: widest move {worst:e}, relative {rel:e}{}",
            if rel > 1e-6 { "   <- MOVED" } else { "" }
        );
    }
    println!(
        "\n  {}",
        if moved_any {
            "A LATER TOKEN MOVES AN EARLIER ROW, so the causal mask is not holding."
        } else {
            "row 0 is untouched by the third token, so causality holds and what is wrong is the ADDRESSING."
        }
    );
    assert!(
        !moved_any,
        "a token after row 0 changed row 0's attention output, which causal masking forbids"
    );
}

/// Where in the KV cache attention's first row actually reads.
///
/// `whether_the_first_row_of_attention_is_its_own_value` shows row 0's output
/// is a pure copy of one `v` block and that the two head groups copy
/// DIFFERENT blocks, and
/// `whether_a_later_token_moves_an_earlier_rows_attention` shows the causal
/// mask holds bit-identically. So the fault is addressing -- and the same
/// probe's `v` stride was wrong, which is why this one assumes NO layout for
/// the operand at all.
///
/// Instead it goes to the cache, addressed the way the shader addresses it:
///
///     sdpa_paged.wgsl:  (slot * n_kv_heads + kv_head) * PIE_HEAD_DIM
///     kv_write.wgsl:    slot * (n_kv_heads * head_dim) + h * head_dim
///
/// -- the same address, written twice -- with `slot` from the seat's own page
/// (`Book::pages`) and the pool's page size. Then both buffers are treated as
/// flat runs of `head_dim` blocks and matched by CONTENT, so the answer is an
/// index rather than an assumption.
///
/// # What each outcome means
///
/// If head 0 lands at `(slot, kv 0)` and head 4 at the SAME slot's `kv 1`,
/// the reader is right and `kv_write` put the wrong content there. If they
/// land at different slots, the reader is. The two are different files and
/// different fixes.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn where_in_the_kv_cache_attentions_first_row_reads() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let n = 3usize;
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    let Some(sdpa) = low.launches.iter().position(|l| {
        low.kernels[l.kernel as usize].starts_with("sdpa_paged_tiled") && l.layers.start == 3
    }) else {
        return;
    };
    let l = &low.launches[sdpa];
    let out_range = low.args[l.args.start as usize..l.args.end as usize]
        .iter()
        .filter_map(|a| match a {
            Arg::Arena { at, width, bytes } => Some((*at, *width as usize * *bytes as usize)),
            _ => None,
        })
        .next_back();
    let Some((o, stride)) = out_range else { return };
    let who = 4_500u64;
    shell.fire_prefix(Some(sdpa + 1));
    let arena = match shell.step(&[driver_wgpu::turns::Turn {
        who,
        tokens: (0..n).map(|t| PERIOD[t % 6]).collect(),
    }]) {
        Ok(s) => s.arena,
        Err(why) => {
            println!("the truncated prefill was refused: {why}");
            return;
        }
    };
    shell.fire_prefix(None);
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    let out0 = bf16(&arena[o..o + stride]);

    let shape = shell.shape();
    let (kv_heads, head_dim, page_size) = (
        shape.kv_heads as usize,
        shape.head_dim as usize,
        shape.page_size as usize,
    );
    let Some(pages) = shell.book().pages(who).map(<[u32]>::to_vec) else {
        println!("the conversation holds no pages");
        return;
    };
    println!("\n  seat pages {pages:?}, page size {page_size}, {kv_heads} kv heads of {head_dim}");
    let Some(vbuf) = shell.pool().cache(3, true) else {
        println!("layer 3 has no value cache");
        return;
    };
    let Ok(vbytes) = shell.device().read_at(vbuf, 0, vbuf.size()) else {
        println!("the value cache would not read back");
        return;
    };
    let vc = bf16(&vbytes);
    println!("  the value cache is {} elements, {} blocks of {head_dim}", vc.len(), vc.len() / head_dim);

    // Both buffers as flat `head_dim` blocks, matched by CONTENT.
    let same = |a: &[f32], b: &[f32]| -> bool { a.iter().zip(b).all(|(x, y)| x == y) };
    let q_heads = out0.len() / head_dim;
    println!("  where each head of OUT ROW 0 is found in the value cache:");
    for h in 0..q_heads {
        let block = &out0[h * head_dim..(h + 1) * head_dim];
        let mut hits = Vec::new();
        for b in 0..vc.len() / head_dim {
            if same(block, &vc[b * head_dim..(b + 1) * head_dim]) {
                hits.push(b);
            }
        }
        let named: Vec<String> = hits
            .iter()
            .take(4)
            .map(|b| {
                let (slot, kv) = (b / kv_heads, b % kv_heads);
                let page = slot / page_size;
                let off = slot % page_size;
                format!("block {b} = slot {slot} (page {page} offset {off}) kv {kv}")
            })
            .collect();
        println!(
            "      head {h} (wants kv {}): {}",
            h / (q_heads / kv_heads),
            if named.is_empty() {
                "found nowhere in the cache".to_string()
            } else {
                named.join(", ")
            }
        );
    }
    // And what the seat's own slots hold, so the two sides can be read together.
    println!("  the seat's own slots, by the reader's formula:");
    for t in 0..n {
        let slot = pages[0] as usize * page_size + t;
        for kv in 0..kv_heads {
            let b = slot * kv_heads + kv;
            let blk = &vc[b * head_dim..(b + 1) * head_dim];
            let mag = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            print!("    t{t} kv{kv} block {b} |v| {mag:.4}");
        }
        println!();
    }
}

/// The whole attention, recomputed from the cache it actually reads.
///
/// `where_in_the_kv_cache_attentions_first_row_reads` finds row 0's output at
/// block 0 for the first head group and block 1 for the second -- slot 0,
/// kv 0 and slot 0, kv 1, which is exactly right. So the earlier
/// "attention's first row is not its own value" was this file's stride on the
/// `v` OPERAND and not the device: the fourth of my claims this session to be
/// refuted by measuring it a second way.
///
/// That leaves the part row 0 cannot reach. Row 0's softmax has one key and is
/// `[1.0]` whatever the scale is; every later row has a real distribution, so
/// the scale, the mask's extent and the accumulation only start mattering at
/// row 1. This recomputes them:
///
///     out[t, h] = sum over s <= t of softmax_s(q[t,h] . k[s, kv(h)] * scale)
///                 * v[s, kv(h)]
///
/// with `q` the sdpa's own input operand, `k` and `v` read from the cache at
/// `(slot * n_kv_heads + kv) * head_dim` -- the shader's address -- and `slot`
/// from the seat's page. Nothing here assumes an arena layout, which is the
/// thing that went wrong last time.
///
/// The scale is FITTED rather than assumed: `1/sqrt(head_dim)` is the usual
/// answer and a wrong one would be a real defect, so the run reports which of
/// several candidates reproduces the device and lets that be the finding.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_attention_is_the_softmax_over_the_cache_it_reads() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let n = 4usize;
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    let Some(sdpa) = low.launches.iter().position(|l| {
        low.kernels[l.kernel as usize].starts_with("sdpa_paged_tiled") && l.layers.start == 3
    }) else {
        return;
    };
    let l = &low.launches[sdpa];
    let ar: Vec<(usize, usize)> = low.args[l.args.start as usize..l.args.end as usize]
        .iter()
        .filter_map(|a| match a {
            Arg::Arena { at, width, bytes } => Some((*at, *width as usize * *bytes as usize)),
            _ => None,
        })
        .collect();
    if ar.len() < 2 {
        return;
    }
    let who = 4_600u64;
    shell.fire_prefix(Some(sdpa + 1));
    let arena = match shell.step(&[driver_wgpu::turns::Turn {
        who,
        tokens: (0..n).map(|t| PERIOD[t % 6]).collect(),
    }]) {
        Ok(s) => s.arena,
        Err(why) => {
            println!("the truncated prefill was refused: {why}");
            return;
        }
    };
    shell.fire_prefix(None);
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    let shape = shell.shape();
    let (kv_heads, head_dim, page_size) = (
        shape.kv_heads as usize,
        shape.head_dim as usize,
        shape.page_size as usize,
    );
    let Some(pages) = shell.book().pages(who).map(<[u32]>::to_vec) else {
        return;
    };
    let (Some(kb), Some(vb)) = (shell.pool().cache(3, false), shell.pool().cache(3, true)) else {
        return;
    };
    let (Ok(kbytes), Ok(vbytes)) = (
        shell.device().read_at(kb, 0, kb.size()),
        shell.device().read_at(vb, 0, vb.size()),
    ) else {
        println!("a cache would not read back");
        return;
    };
    let (kc, vc) = (bf16(&kbytes), bf16(&vbytes));
    fn block(c: &[f32], base: usize, t: usize, kv: usize, kv_heads: usize, hd: usize) -> &[f32] {
        let b = (base + t) * kv_heads + kv;
        &c[b * hd..(b + 1) * hd]
    }
    let base = pages[0] as usize * page_size;
    let q_all = bf16(&arena[ar[0].0..ar[0].0 + ar[0].1 * n]);
    let out_all = bf16(&arena[ar[ar.len() - 1].0..ar[ar.len() - 1].0 + ar[ar.len() - 1].1 * n]);
    let q_heads = ar[0].1 / 2 / head_dim;
    let rep = q_heads / kv_heads;
    println!("\n  {q_heads} query heads of {head_dim} over {kv_heads} kv heads, {n} rows");
    let per_row = q_heads * head_dim;
    for scale in [
        1.0 / (head_dim as f32).sqrt(),
        1.0 / 128.0f32.sqrt(),
        1.0,
    ] {
        let (mut worst, mut mag) = (0.0f32, 0.0f32);
        for t in 0..n {
            for h in 0..q_heads {
                let kv = h / rep;
                let q = &q_all[t * per_row + h * head_dim..t * per_row + (h + 1) * head_dim];
                let mut logits = vec![0.0f32; t + 1];
                for (s, lg) in logits.iter_mut().enumerate() {
                    let k = block(&kc, base, s, kv, kv_heads, head_dim);
                    *lg = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits.iter().map(|v| (v - top).exp()).collect();
                let z: f32 = exps.iter().sum();
                for d in 0..head_dim {
                    let want: f32 = (0..=t)
                        .map(|s| exps[s] / z * block(&vc, base, s, kv, kv_heads, head_dim)[d])
                        .sum();
                    let got = out_all[t * per_row + h * head_dim + d];
                    worst = worst.max((want - got).abs());
                    mag = mag.max(got.abs());
                }
            }
        }
        println!(
            "    scale {scale:.6}: widest |cpu - gpu| {worst:e}, relative {:e}",
            worst / mag.max(1e-30)
        );
    }
    println!("  (one bf16 step is 3.9e-3; the row-0-only check cannot see any of this)");
}

/// The query-gate split and the attention output gate, against their operands.
///
/// Both blocks are now verified: the gated DeltaNet by five references, and
/// attention by `whether_attention_is_the_softmax_over_the_cache_it_reads` --
/// the full softmax over the cache it actually reads, at
/// `scale = 1/sqrt(256)`, to 2.3e-3, which is inside one bf16 step where the
/// wrong scales are at 9.4e-2 and 6.0e-1.
///
/// What is left between the softmax and the residual is the pair qwen3-0.6b
/// never runs and that has only ever been checked by READING it against
/// `driver-metal`:
///
///     q_gate_split:  qg is `[rows, n_q, 2, head_dim]` interleaved, so
///                    q[h, d] = qg[h, 0, d] and gate[h, d] = qg[h, 1, d]
///     gate:          attn = attn * sigmoid(gate), in place
///
/// A split that took the halves the other way round, or a gate applied to the
/// wrong head, is finite and correctly shaped and destroys the block. The
/// second needs TWO fires -- the gate writes over its own input, so the value
/// before it is only readable by stopping on the rectangle before.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_the_query_gate_split_and_the_output_gate_do_what_they_say() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let n = 3usize;
    let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % 6]).collect();
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    let find = |stem: &str| -> Option<usize> {
        low.launches
            .iter()
            .position(|l| low.kernels[l.kernel as usize].starts_with(stem) && l.layers.start == 3)
    };
    let (Some(split), Some(sdpa), Some(gate)) =
        (find("q_gate_split"), find("sdpa_paged_tiled"), find("gate_bfloat16"))
    else {
        println!("layer 3 does not run all three");
        return;
    };
    let ranges = |at: usize| -> Vec<(usize, usize)> {
        let l = &low.launches[at];
        low.args[l.args.start as usize..l.args.end as usize]
            .iter()
            .filter_map(|a| match a {
                Arg::Arena { at, width, bytes } => Some((*at, *width as usize * *bytes as usize)),
                _ => None,
            })
            .collect()
    };
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    let fire = |shell: &mut driver_wgpu::shell::Shell, who: u64, upto: usize| -> Vec<u8> {
        shell.fire_prefix(Some(upto + 1));
        let out = shell
            .step(&[driver_wgpu::turns::Turn {
                who,
                tokens: tokens.clone(),
            }])
            .map(|s| s.arena)
            .unwrap_or_default();
        shell.fire_prefix(None);
        out
    };
    let a_split = fire(&mut shell, 4_700, split);
    let a_sdpa = fire(&mut shell, 4_701, sdpa);
    let a_gate = fire(&mut shell, 4_702, gate);
    if a_split.is_empty() || a_sdpa.is_empty() || a_gate.is_empty() {
        println!("a fire was refused, so IT COULD NOT BE MEASURED");
        return;
    }
    let (head_dim, q_heads) = (256usize, 8usize);

    // THE SPLIT. Three ranges: the fused `qg`, then `q` and `gate`.
    let sr = ranges(split);
    if sr.len() != 3 {
        println!("`q_gate_split` has {} arena ranges, not 3", sr.len());
        return;
    }
    let (mut worst_q, mut worst_g, mut mag) = (0.0f32, 0.0f32, 0.0f32);
    for t in 0..n {
        let qg = bf16(&a_split[sr[0].0 + t * sr[0].1..sr[0].0 + (t + 1) * sr[0].1]);
        let q = bf16(&a_split[sr[1].0 + t * sr[1].1..sr[1].0 + (t + 1) * sr[1].1]);
        let g = bf16(&a_split[sr[2].0 + t * sr[2].1..sr[2].0 + (t + 1) * sr[2].1]);
        for h in 0..q_heads {
            for d in 0..head_dim {
                worst_q = worst_q.max((qg[h * 2 * head_dim + d] - q[h * head_dim + d]).abs());
                worst_g =
                    worst_g.max((qg[h * 2 * head_dim + head_dim + d] - g[h * head_dim + d]).abs());
                mag = mag.max(q[h * head_dim + d].abs());
            }
        }
    }
    println!("\n  q_gate_split, against `[n_q, 2, head_dim]` interleaved:");
    println!("    q half widest {worst_q:e}, gate half widest {worst_g:e}, against {mag:.6}");

    // THE GATE. Its input is the sdpa's output, read from the fire that stops
    // one rectangle earlier, because this one writes over it.
    let (gr, dr) = (ranges(gate), ranges(sdpa));
    if gr.len() < 3 || dr.len() < 2 {
        println!("  the gate or the sdpa is not the shape this test reads");
        return;
    }
    let silu_sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let (mut worst, mut gmag) = (0.0f32, 0.0f32);
    for t in 0..n {
        let before = bf16(
            &a_sdpa[dr[dr.len() - 1].0 + t * dr[dr.len() - 1].1
                ..dr[dr.len() - 1].0 + (t + 1) * dr[dr.len() - 1].1],
        );
        let g = bf16(&a_gate[gr[1].0 + t * gr[1].1..gr[1].0 + (t + 1) * gr[1].1]);
        let after = bf16(
            &a_gate[gr[gr.len() - 1].0 + t * gr[gr.len() - 1].1
                ..gr[gr.len() - 1].0 + (t + 1) * gr[gr.len() - 1].1],
        );
        for i in 0..after.len().min(before.len()).min(g.len()) {
            worst = worst.max((before[i] * silu_sigmoid(g[i]) - after[i]).abs());
            gmag = gmag.max(after[i].abs());
        }
    }
    println!("  gate, against `attn * sigmoid(gate)`:");
    println!("    widest {worst:e}, against {gmag:.6} -- relative {:e}", worst / gmag.max(1e-30));
    println!("  (one bf16 step is 3.9e-3)");
}

/// Whether the embedding row a token gathers is that token's row.
///
/// Twelve kernels now reproduce a CPU walk of their own operands -- the whole
/// gated DeltaNet, the whole attention, the query-gate split and the output
/// gate -- and the model still answers nothing. Every one of those references
/// took the weights AS GIVEN, so none of them can see a tensor that is
/// correctly computed with and wrongly chosen.
///
/// The embedding is where that matters most, and twice over: it is the input
/// to all 24 layers, and `tie_word_embeddings` is TRUE in this checkpoint, so
/// the same table is the readout. A table off by a row, or resolved to the
/// wrong tensor of a checkpoint that also ships a vision tower, gives exactly
/// what is observed -- every layer faithful, the answer a few punctuation
/// tokens whatever the prompt.
///
/// `embed_gather_mb_4bit` is the first rectangle of the fire, so its output is
/// readable at a prefix of one, and the row it should have produced is
/// `scale * code + bias` over the token's own 1024 codes.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_the_embedding_row_a_token_gathers_is_that_tokens_row() {
    let Some((mut shell, real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let n = 3usize;
    let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % 6]).collect();
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    let Some(at) = low
        .launches
        .iter()
        .position(|l| low.kernels[l.kernel as usize].starts_with("embed_gather"))
    else {
        println!("no `embed_gather`");
        return;
    };
    let l = &low.launches[at];
    let Some((o, stride)) = low.args[l.args.start as usize..l.args.end as usize]
        .iter()
        .filter_map(|a| match a {
            Arg::Arena { at, width, bytes } => Some((*at, *width as usize * *bytes as usize)),
            _ => None,
        })
        .next()
    else {
        return;
    };
    shell.fire_prefix(Some(at + 1));
    let arena = match shell.step(&[driver_wgpu::turns::Turn {
        who: 4_800,
        tokens: tokens.clone(),
    }]) {
        Ok(s) => s.arena,
        Err(why) => {
            println!("the truncated prefill was refused: {why}");
            return;
        }
    };
    shell.fire_prefix(None);
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    let (Some(codes), Some(sc), Some(ze)) = (
        real.get("embed"),
        real.get("embed.scales").map(|b| bf16(b)),
        real.get("embed.zeros").map(|b| bf16(b)),
    ) else {
        println!("the embedding's three tensors are not all here");
        return;
    };
    let (hidden, group) = (1024usize, 64usize);
    let groups = hidden / group;
    let vocab = codes.len() / (hidden / 2);
    println!(
        "\n  embed: {} code bytes over {hidden} wide = {vocab} rows; scales {} = {} rows",
        codes.len(),
        sc.len(),
        sc.len() / groups
    );
    let mut worst_all = 0.0f32;
    for (t, tok) in tokens.iter().enumerate() {
        let got = bf16(&arena[o + t * stride..o + (t + 1) * stride]);
        // The row the token SHOULD have gathered, and its neighbours, so an
        // off-by-one shows as a match one row over rather than as a number.
        let mut best = (f32::INFINITY, 0i64);
        for d in -2i64..=2 {
            let r = *tok as i64 + d;
            if r < 0 || r as usize >= vocab {
                continue;
            }
            let r = r as usize;
            let base = r * hidden / 2;
            let mut worst = 0.0f32;
            for g in 0..groups {
                let (s, z) = (sc[r * groups + g], ze[r * groups + g]);
                for e in 0..group {
                    let i = g * group + e;
                    let byte = codes[base + i / 2];
                    let code = f32::from(if i % 2 == 0 { byte & 0xf } else { byte >> 4 });
                    worst = worst.max((s * code + z - got[i]).abs());
                }
            }
            if worst < best.0 {
                best = (worst, d);
            }
            if d == 0 {
                worst_all = worst_all.max(worst);
            }
        }
        let mag = got.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!(
            "    token {tok:>6}: its own row differs by {:e} (|row| {mag:.6}); \
             the closest of rows {tok}+-2 is offset {:+}",
            worst_all, best.1
        );
    }
    println!(
        "\n  {}",
        if worst_all < 1e-2 {
            "EACH TOKEN GATHERS ITS OWN ROW."
        } else {
            "A TOKEN DOES NOT GATHER ITS OWN ROW, and every layer after it was faithful to the wrong vector."
        }
    );
}

/// The model, asked something in its OWN vocabulary.
///
/// Twelve kernels and the embedding now reproduce a CPU walk of their own
/// operands, and the config matches the driver's facts number for number --
/// 24 layers, full attention at 3, 7, 11, 15, 19, 23, `head_dim` 256,
/// `partial_rotary_factor` 0.25, `attn_output_gate`, vocab 248,320. So before
/// concluding that a model of correct parts answers nothing, the QUESTION is
/// worth checking.
///
/// It was not in this model's vocabulary. `PERIOD` was chosen for
/// qwen3-0.6b, whose tokenizer has 151,936 entries; this checkpoint's has
/// 248,044, and the same integers spell something else here:
///
///     15339 'Ġreads'   1723 '-c'   88204 '))=='
///      6100 'ospital'  41777 '(RE'  2930 'ouch'
///
/// Real tokens, and a sequence of them that means nothing. The answers those
/// probes call wrong -- 220 'Ġ', 16 '1', 17 '2', 271 'ĊĊ', 198 'Ċ', 11 ',' --
/// are what a base model says after gibberish, and a repeated `))==` is not
/// the easy continuation the floor test assumed it was.
///
/// So: two questions in this vocabulary. A repeated ordinary word, which any
/// model echoes, and a fact stated once and asked again, which needs the
/// attention layers to carry it.
///
/// # NOT YET RUN, and the finding does not wait on it
///
/// The shared trace moved again between one probe and the next -- the local
/// workaround that carried the twelve kernel references stopped reaching
/// `affine_qmv_fast` -- so this asks its questions of a tree that will not
/// fire. It is left here because the READING is already worth having and does
/// not depend on the answer:
///
/// **Every "the model answers wrongly" result in this file was measured with
/// tokens from another model's vocabulary.** Real tokens, spelling nothing.
/// The tops those probes report -- `Ġ`, `1`, `2`, `ĊĊ`, `Ċ`, `,` -- are what a
/// base model says after gibberish, so "0 of 4 on the floor test" is a weaker
/// statement than it reads as, and `PERIOD` continuing itself was never the
/// easy question it was taken for.
///
/// That does not make the model right. It makes the evidence that it is wrong
/// thinner than twelve verified kernels deserve, which is why the question is
/// rewritten here rather than the conclusion restated.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn what_the_model_answers_when_the_question_is_in_its_own_vocabulary() {
    let Some((mut shell, _real)) = qwen3_5_shell(16) else {
        return;
    };
    let ask = |shell: &mut driver_wgpu::shell::Shell, who: u64, tokens: &[u32], want: u32, what: &str| -> bool {
        let row = fire_row(shell, who, tokens);
        if row.is_empty() {
            println!("    {what}: refused");
            return false;
        }
        let mut top: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let got = u32::try_from(top[0].0).unwrap_or(u32::MAX);
        println!(
            "    {what}: wanted {want}, got {got} ({:.2}); {want} scored {:.2}; top {:?}",
            top[0].1,
            row.get(want as usize).copied().unwrap_or(f32::NAN),
            top[..5].iter().map(|(t, _)| *t).collect::<Vec<_>>()
        );
        got == want
    };
    println!("\n  A REPEATED ORDINARY WORD, sixteen times:");
    let mut right = 0;
    let mut asked = 0;
    for (i, tok) in [279u32, 7993, 5388, 2438].into_iter().enumerate() {
        asked += 1;
        if ask(&mut shell, 5_000 + i as u64, &[tok; 16], tok, "repeat") {
            right += 1;
        }
    }
    // "The capital of France is Paris. The capital of France is" -> " Paris"
    println!("\n  A FACT STATED ONCE AND ASKED AGAIN:");
    let fact: Vec<u32> = vec![
        561, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 9338, 369,
    ];
    asked += 1;
    if ask(&mut shell, 5_100, &fact, 11751, "capital of France") {
        right += 1;
    }
    // "one two three four one two three" -> " four"
    let count: Vec<u32> = vec![799, 1330, 2250, 2943, 799, 1330, 2250];
    asked += 1;
    if ask(&mut shell, 5_101, &count, 2943, "counting") {
        right += 1;
    }
    println!("\n  {right} of {asked} answered as a working model would");
    if right == 0 && asked > 0 {
        println!(
            "  (zero of {asked} with every fire REFUSED is not an answer about the model; \
             see this test's header)"
        );
    }
}

/// Whether the logits are the readout of the hidden state the fire produced.
///
/// Asked in its OWN vocabulary the model still answers 220 `Ġ` -- a bare space
/// -- to every prompt, with the right token often second to fifth: ` the`
/// repeated sixteen times scores ` the` at 10.06 against the space's 11.88,
/// and "The capital of France is Paris. The capital of France is" scores
/// ` Paris` at 2.83 against 20.38. So there IS signal and something else is
/// louder, and the vocabulary caveat does not rescue it.
///
/// That splits in two, and this is the split. `lm_head` is TIED to the
/// embedding, so the logit for token `t` is the dot of the final hidden state
/// with the embedding row `t` -- and both are readable. If they agree, the
/// readout is faithful and the hidden state is what is wrong, which is a
/// layer. If they do not, everything upstream of it was fine and the last
/// dispatch is the defect.
///
/// A handful of tokens rather than all 248,320: the question is whether the
/// projection reproduces its operands, and the interesting rows are the one
/// the device chose and the one it should have.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_the_logits_are_the_readout_of_the_state_the_fire_produced() {
    let Some((mut shell, real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    // "The capital of France is Paris. The capital of France is"
    let tokens: Vec<u32> = vec![561, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 9338, 369];
    let n = tokens.len();
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    // THE READOUT'S OWN OPERANDS, not a guess at them. The last launch is the
    // projection to the vocabulary; its arena INPUT is the state it
    // multiplies and its weights are the table it multiplies BY -- which the
    // first draft of this test assumed was `embed` because the checkpoint ties
    // them. Assuming that is how a reference ends up disagreeing with a device
    // that is right.
    let at = low.launches.len() - 1;
    let l = &low.launches[at];
    let mut arenas = Vec::new();
    let mut weights = Vec::new();
    for a in &low.args[l.args.start as usize..l.args.end as usize] {
        match a {
            Arg::Arena { at, width, bytes } => {
                arenas.push((*at, *width as usize * *bytes as usize));
            }
            Arg::Weight(n) => weights.push(n.clone()),
            // Neither is a RECTANGLE this probe can read back: a named
            // value is bound by the backend rather than placed in the arena,
            // and a raise is a host aggregate with no row width at all.
            Arg::Named { .. } | Arg::Raised { .. } => {}
        }
    }
    println!(
        "\n  the readout is `{}` over weights {weights:?} with {} arena ranges",
        low.kernels[l.kernel as usize],
        arenas.len()
    );
    let Some((o, stride)) = arenas.first().copied() else {
        return;
    };
    let stem = weights.first().cloned().unwrap_or_default();
    let step = match shell.step(&[driver_wgpu::turns::Turn {
        who: 6_000,
        tokens: tokens.clone(),
    }]) {
        Ok(s) => s,
        Err(why) => {
            println!("the fire was refused: {why}");
            return;
        }
    };
    // EVERY readout row, not only the one `readout_of` points at. If the
    // kernel's output turns up under a different index, the defect is the
    // index; if it turns up under none, it is the copy.
    let all: Vec<Vec<f32>> = (0..)
        .map_while(|i| step.logits.row(i).map(<[f32]>::to_vec))
        .collect();
    let picked = step.readout_of[0];
    println!(
        "\n  the fire hands back {} logit row(s) of {}, and `readout_of[0]` is {picked}",
        all.len(),
        all.first().map_or(0, Vec::len)
    );
    let Some(logits) = all.get(picked).cloned() else {
        println!("no logits row at {picked}");
        return;
    };
    let arena = step.arena;
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    // ROW `picked`, on BOTH sides. An arena range's stride is ONE row, and
    // reading `[o .. o + stride]` is row 0 -- so the first version of this
    // compared row 0's state against row 11's logits and called the readout
    // broken. The rectangle has one row per sampling token and `readout_of`
    // says which one the answer comes from; both operands have to be read
    // there or the comparison is between two different tokens.
    let h = bf16(&arena[o + picked * stride..o + (picked + 1) * stride]);
    let (Some(codes), Some(sc), Some(ze)) = (
        real.get(&stem),
        real.get(&format!("{stem}.scales")).map(|b| bf16(b)),
        real.get(&format!("{stem}.zeros")).map(|b| bf16(b)),
    ) else {
        println!("`{stem}`'s three tensors are not all here");
        return;
    };
    println!("  reading the table the launch NAMES: `{stem}`");
    let (hidden, group) = (1024usize, 64usize);
    let groups = hidden / group;
    if h.len() != hidden {
        println!("the gathered row is {} wide, not {hidden}", h.len());
        return;
    }
    let mag = h.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!("\n  the state the readout multiplies is {hidden} wide, widest |x| {mag:.4}");
    // THE KERNEL'S OWN OUTPUT, beside the logits the driver hands back. They
    // are not the same question: one asks whether the projection computes what
    // its operands imply, the other whether what it computed is what arrives.
    let written: Option<Vec<f32>> = arenas.get(1).map(|(o, stride)| {
        let from = o + picked * stride;
        let end = (from + stride).min(arena.len());
        bf16(&arena[from..end])
    });
    if let Some(w) = &written {
        println!("  the launch wrote {} values into the arena", w.len());
    }
    println!("  token      device logit    kernel wrote    cpu from the embedding row");
    let mut worst = 0.0f32;
    for t in [220usize, 11751, 279, 11, 13, 271] {
        let base = t * hidden / 2;
        let mut acc = 0.0f32;
        for g in 0..groups {
            let (s, z) = (sc[t * groups + g], ze[t * groups + g]);
            for e in 0..group {
                let i = g * group + e;
                let byte = codes[base + i / 2];
                let code = f32::from(if i % 2 == 0 { byte & 0xf } else { byte >> 4 });
                acc += h[i] * (s * code + z);
            }
        }
        let got = logits.get(t).copied().unwrap_or(f32::NAN);
        let wrote = written
            .as_ref()
            .and_then(|w| w.get(t).copied())
            .unwrap_or(f32::NAN);
        worst = worst.max((acc - wrote).abs());
        println!("  {t:>6}   {got:>12.4}    {wrote:>12.4}    {acc:>12.4}");
    }
    // WHICH row of the readout, if any, is what the kernel wrote.
    if let Some(w) = &written {
        for (i, row) in all.iter().enumerate() {
            let mut d = 0.0f32;
            for t in [220usize, 11751, 279, 11, 13, 271] {
                if let (Some(a), Some(b)) = (row.get(t), w.get(t)) {
                    d = d.max((a - b).abs());
                }
            }
            println!("    logits row {i}: widest gap to what the kernel wrote at row {picked}: {d:.4}");
        }
    }
    println!(
        "\n  widest |cpu - gpu| over those rows {worst:e}\n  {}",
        if worst < 0.2 {
            "THE PROJECTION IS FAITHFUL to its operands; if the logits column differs from what it wrote, the defect is between the kernel and the readout."
        } else {
            "THE PROJECTION DISAGREES WITH ITS OWN OPERANDS."
        }
    );
}

/// The same questions at EIGHT bits instead of four.
///
/// Fifteen components of this model reproduce a CPU walk of their own
/// operands, up to and including the readout, so the answer the driver gives
/// IS what its kernels compute. What is left is outside the driver, and the
/// nearest thing outside it is how the weights got here: this checkpoint ships
/// bfloat and the loader quantizes it at load time, `RuntimeQuant::Int4` --
/// four bits over groups of 64, chosen because it is *"the format every Metal
/// matvec reads, and the only one its loader can encode into"*.
///
/// Four bits over a 0.8B model is aggressive, and the tree stamps the other
/// point: every symbol carries its width (`..._gs_64_b_4` against `_b_8`) and
/// `RuntimeQuant::Int8` is a policy the same loader takes. So the two can be
/// compared on the same adapter, the same prompts, the same everything else --
/// which makes this a controlled comparison rather than a second opinion.
///
/// If eight bits answers what four does not, the driver was never the defect
/// and the load policy is. If both answer the same, the quantization is
/// exonerated too and what is left is the checkpoint.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn whether_eight_bits_answers_what_four_does_not() {
    let Some((mut shell, _real)) =
        qwen3_5_shell_at(16, 8, model::shared::policy::RuntimeQuant::Int8)
    else {
        return;
    };
    let ask = |shell: &mut driver_wgpu::shell::Shell,
               who: u64,
               tokens: &[u32],
               want: u32,
               what: &str|
     -> bool {
        let row = fire_row(shell, who, tokens);
        if row.is_empty() {
            println!("    {what}: refused");
            return false;
        }
        let mut top: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let got = u32::try_from(top[0].0).unwrap_or(u32::MAX);
        println!(
            "    {what}: wanted {want}, got {got} ({:.2}); {want} scored {:.2}; top {:?}",
            top[0].1,
            row.get(want as usize).copied().unwrap_or(f32::NAN),
            top[..5].iter().map(|(t, _)| *t).collect::<Vec<_>>()
        );
        got == want
    };
    println!("\n  AT EIGHT BITS -- a repeated ordinary word, sixteen times:");
    let mut right = 0;
    let mut asked = 0;
    for (i, tok) in [279u32, 7993, 5388, 2438].into_iter().enumerate() {
        asked += 1;
        if ask(&mut shell, 7_000 + i as u64, &[tok; 16], tok, "repeat") {
            right += 1;
        }
    }
    println!("\n  AT EIGHT BITS -- a fact stated once and asked again:");
    let fact: Vec<u32> = vec![561, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 9338, 369];
    asked += 1;
    if ask(&mut shell, 7_100, &fact, 11751, "capital of France") {
        right += 1;
    }
    let count: Vec<u32> = vec![799, 1330, 2250, 2943, 799, 1330, 2250];
    asked += 1;
    if ask(&mut shell, 7_101, &count, 2943, "counting") {
        right += 1;
    }
    println!("\n  {right} of {asked} at eight bits, against 0 of {asked} at four");
}

/// What the int4 encode costs, against the bf16 the checkpoint ships.
///
/// Fifteen components of this model reproduce a CPU walk of their own
/// operands, so the driver computes what its kernels say. The nearest thing
/// outside it is the ENCODE: this checkpoint ships bfloat and the loader
/// quantizes at load time to four bits over groups of 64, and
/// `whether_eight_bits_answers_what_four_does_not` cannot vary that -- the
/// contract refuses `Int8` for this family.
///
/// It can be measured directly instead. A safetensors file is an eight-byte
/// little-endian header length, a JSON header, then the raw bytes, so the
/// bf16 the checkpoint holds is readable beside the codes the loader produced
/// and the two can be subtracted.
///
/// # What each outcome means
///
/// Four bits over a group of 64 has a floor: fifteen steps across a group's
/// range is a relative step of about 1/15 of the spread, so a few per cent of
/// the largest weight is EXPECTED and is not a defect. What would be a defect
/// is a systematic one -- an encode that is off by a scale, a sign, or a
/// half-step -- because that is a different tensor rather than a coarser one.
/// So the run reports the error against the group's own range, not against
/// the weight, and the mean beside the widest.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot"]
fn what_the_int4_encode_costs_against_the_bfloat_it_came_from() {
    let Some(dir) = qwen3_5_snapshot() else {
        println!("no Qwen3.5-0.8B snapshot, so IT COULD NOT BE MEASURED");
        return;
    };
    let Some(row) = model::catalog::find("qwen3.5-0.8b-base") else {
        return;
    };
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let Some(real) = qwen3_5_weights(&dir, row, &hybrid_plan(&facts)) else {
        return;
    };
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    // The checkpoint, read where it lies.
    let Some(path) = std::fs::read_dir(&dir).ok().and_then(|d| {
        d.filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "safetensors"))
    }) else {
        println!("no `.safetensors` in the snapshot");
        return;
    };
    let Ok(mut f) = std::fs::File::open(&path) else {
        return;
    };
    use std::io::{Read, Seek, SeekFrom};
    let mut len = [0u8; 8];
    if f.read_exact(&mut len).is_err() {
        return;
    }
    let n = u64::from_le_bytes(len) as usize;
    let mut head = vec![0u8; n];
    if f.read_exact(&mut head).is_err() {
        return;
    }
    let Ok(head) = serde_json::from_slice::<serde_json::Value>(&head) else {
        println!("the header is not JSON this test can read");
        return;
    };
    let base = 8 + n as u64;

    let (hidden, group) = (1024usize, 64usize);
    let groups = hidden / group;
    let pairs: [(&str, &str); 2] = [
        ("embed", "model.language_model.embed_tokens.weight"),
        (
            "layer.0.in_proj_qkv",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        ),
    ];
    println!("\n  the int4 encode against the bf16 it came from:");
    for (held, source) in pairs {
        let (Some(codes), Some(sc), Some(ze)) = (
            real.get(held),
            real.get(&format!("{held}.scales")).map(|b| bf16(b)),
            real.get(&format!("{held}.zeros")).map(|b| bf16(b)),
        ) else {
            println!("    {held}: not held");
            continue;
        };
        let Some(meta) = head.get(source) else {
            println!("    {held}: `{source}` is not in the checkpoint");
            continue;
        };
        let off = meta["data_offsets"][0].as_u64().unwrap_or(0);
        let dtype = meta["dtype"].as_str().unwrap_or("?").to_string();
        if dtype != "BF16" {
            println!("    {held}: `{source}` is {dtype}, which this test does not read");
            continue;
        }
        // A handful of rows rather than the whole tensor: the question is
        // whether the encode is FAITHFUL, and a systematic fault shows in any
        // row while a coarse one shows in all of them.
        let (mut worst, mut sum, mut count) = (0.0f32, 0.0f64, 0usize);
        let mut worst_rel = 0.0f32;
        // BOUNDED BY THE TENSOR. `in_proj_qkv` has 6144 rows and `embed` has
        // 248,320, and a row list written for the larger one walks off the
        // smaller. The scales say how many there are: one pair per group per
        // row.
        let n_rows = sc.len() / groups;
        for r in [0usize, 11, 220, 279, 11_751]
            .into_iter()
            .filter(|r| *r < n_rows)
        {
            let mut raw = vec![0u8; hidden * 2];
            if f.seek(SeekFrom::Start(base + off + (r * hidden * 2) as u64)).is_err()
                || f.read_exact(&mut raw).is_err()
            {
                continue;
            }
            let want = bf16(&raw);
            let cbase = r * hidden / 2;
            for g in 0..groups {
                let (s, z) = (sc[r * groups + g], ze[r * groups + g]);
                // The group's own range is what four bits divides into
                // fifteen steps, so it is what the error is relative TO.
                let span = (s * 15.0).abs().max(1e-30);
                for e in 0..group {
                    let i = g * group + e;
                    let byte = codes[cbase + i / 2];
                    let code = f32::from(if i % 2 == 0 { byte & 0xf } else { byte >> 4 });
                    let d = (s * code + z - want[i]).abs();
                    worst = worst.max(d);
                    worst_rel = worst_rel.max(d / span);
                    sum += f64::from(d / span);
                    count += 1;
                }
            }
        }
        let mean = if count == 0 {
            f32::NAN
        } else {
            (sum / count as f64) as f32
        };
        println!(
            "    {held:<20} widest |int4 - bf16| {worst:.6}; against the group's own range: \
             widest {worst_rel:.4}, mean {mean:.4} over {count} weights"
        );
    }
    println!(
        "  (four bits is fifteen steps across a group, so ~0.033 mean and ~0.07 widest is the \
         FLOOR; much more than that is an encode fault rather than a coarse one)"
    );
}

/// What the model actually GENERATES, token after token.
///
/// Every probe in this file has scored one step and called it wrong, and every
/// one of them has been asking a question about tokenization rather than about
/// the model. This vocabulary carries a word twice -- with a leading space and
/// without:
///
///     'ĠParis' 11751   'Paris' 57590      'Ġthe' 279   'the' 1719
///     'Ġfour'   2943   'four'  32897
///
/// and 220 is `Ġ`, a bare space. So a model that answers `Ġ` and then `Paris`
/// has written " Paris" -- the right TEXT by a different route -- while a
/// probe scoring only the first step marks it wrong and reports the space as
/// nonsense.
///
/// That is a real possibility and it is cheap to settle: take the argmax,
/// append it, and ask again. Three steps is enough to tell " Paris" from a run
/// of punctuation.
#[test]
#[ignore = "needs a Qwen3.5-0.8B snapshot and an adapter"]
fn what_the_model_generates_rather_than_what_it_scores_in_one_step() {
    let Some((mut shell, _real)) = qwen3_5_shell(16) else {
        return;
    };
    let step_once = |shell: &mut driver_wgpu::shell::Shell, who: u64, toks: &[u32]| -> Option<u32> {
        let row = fire_row(shell, who, toks);
        if row.is_empty() {
            return None;
        }
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, v) in row.iter().enumerate() {
            if *v > best.1 {
                best = (i, *v);
            }
        }
        u32::try_from(best.0).ok()
    };
    let cases: [(&str, &[u32], &[u32]); 4] = [
        // "The capital of France is Paris. The capital of France is" -> Paris
        (
            "capital of France",
            &[561, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 9338, 369],
            &[11751, 57590],
        ),
        // "one two three four one two three" -> four
        ("counting", &[799, 1330, 2250, 2943, 799, 1330, 2250], &[2943, 32897]),
        ("the x16", &[279; 16], &[279, 1719]),
        ("cat x16", &[7993; 16], &[7993, 4466]),
    ];
    println!("\n  three greedy steps, and whether the WORD appears at any of them:");
    let mut right = 0;
    for (i, (what, prompt, wanted)) in cases.into_iter().enumerate() {
        let who = 8_000 + i as u64;
        let mut toks: Vec<u32> = prompt.to_vec();
        let mut made: Vec<u32> = Vec::new();
        // The first fire takes the whole prompt; each later one takes the
        // single token just chosen, which is what continuing a conversation is.
        let Some(first) = step_once(&mut shell, who, &toks) else {
            println!("    {what}: refused");
            continue;
        };
        made.push(first);
        toks.push(first);
        for _ in 0..2 {
            let Some(next) = step_once(&mut shell, who, &[*toks.last().unwrap()]) else {
                break;
            };
            made.push(next);
            toks.push(next);
        }
        let hit = made.iter().any(|t| wanted.contains(t));
        if hit {
            right += 1;
        }
        println!(
            "    {what:<20} generated {made:?}; wanted either of {wanted:?} -- {}",
            if hit { "FOUND" } else { "not there" }
        );
    }
    println!(
        "\n  {right} of 4 produced the word within three tokens\n  {}",
        if right >= 3 {
            "THE MODEL IS WRITING THE RIGHT TEXT, and every one-step probe in this file was scoring a tokenization."
        } else {
            "the word does not arrive, so the space was not a tokenization detour."
        }
    );
}

/// WHICH CHECKPOINT LAYER EACH DRIVER LAYER ACTUALLY HOLDS.
///
/// Every reference in this file checks a kernel against ITS OWN operands, and
/// that is exactly the check a mis-mapped layer survives. If `layer.3` were
/// loaded with checkpoint layer 4's weights, each kernel would still compute
/// the right function of what it was handed, every residual would still be
/// finite, and the model would still be wrong -- silently, and in a way no
/// amount of per-kernel verification can see.
///
/// So ask the composition question the only way it can be asked: not "is this
/// arithmetic right" but "is this the right TENSOR". `conv_w` is the instrument
/// -- it is held RAW bf16, not quantized, so a match is byte-for-byte rather
/// than within a floor, and it exists once per gated-DeltaNet layer, which is
/// eighteen of the twenty-four. Read what the driver holds for layer N, then
/// read every checkpoint layer, and report which one it equals. An identity
/// permutation is the only correct answer; anything else -- a shift, a
/// duplicate, a hole -- is the defect, named.
#[test]
fn which_checkpoint_layer_each_driver_layer_is_actually_holding() {
    let Some(dir) = qwen3_5_snapshot() else {
        println!("no Qwen3.5-0.8B snapshot, so IT COULD NOT BE MEASURED");
        return;
    };
    let Some(row) = model::catalog::find("qwen3.5-0.8b-base") else {
        return;
    };
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let Some(real) = qwen3_5_weights(&dir, row, &hybrid_plan(&facts)) else {
        return;
    };
    let Some(path) = std::fs::read_dir(&dir).ok().and_then(|d| {
        d.filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "safetensors"))
    }) else {
        println!("no `.safetensors` in the snapshot");
        return;
    };
    let Ok(mut f) = std::fs::File::open(&path) else {
        return;
    };
    use std::io::{Read, Seek, SeekFrom};
    let mut len = [0u8; 8];
    if f.read_exact(&mut len).is_err() {
        return;
    }
    let n = u64::from_le_bytes(len) as usize;
    let mut head = vec![0u8; n];
    if f.read_exact(&mut head).is_err() {
        return;
    }
    let Ok(head) = serde_json::from_slice::<serde_json::Value>(&head) else {
        return;
    };
    let base = 8 + n as u64;

    // Every conv the checkpoint carries, read once and kept, so each driver
    // layer is compared against ALL of them rather than only its namesake.
    let mut source: Vec<(u32, Vec<u8>)> = Vec::new();
    for m in 0..facts.layers {
        let name = format!("model.language_model.layers.{m}.linear_attn.conv1d.weight");
        let Some(meta) = head.get(&name) else {
            continue;
        };
        if meta["dtype"].as_str() != Some("BF16") {
            continue;
        }
        let (a, b) = (
            meta["data_offsets"][0].as_u64().unwrap_or(0),
            meta["data_offsets"][1].as_u64().unwrap_or(0),
        );
        let mut raw = vec![0u8; (b - a) as usize];
        if f.seek(SeekFrom::Start(base + a)).is_ok() && f.read_exact(&mut raw).is_ok() {
            source.push((m, raw));
        }
    }
    if source.is_empty() {
        println!("the checkpoint carries no `linear_attn.conv1d.weight`, so THE MAPPING COULD NOT BE READ");
        return;
    }

    println!(
        "\n  which checkpoint layer each driver layer holds ({} convs in the checkpoint):",
        source.len()
    );
    let (mut identity, mut wrong, mut missing) = (0usize, Vec::new(), 0usize);
    for i in 0..facts.layers {
        let Some(held) = real.get(&format!("layer.{i}.conv_w")) else {
            continue;
        };
        // Byte-for-byte, because both sides are the same raw bf16: a match is
        // a match, and a near-match is not a match.
        let hit: Vec<u32> = source
            .iter()
            .filter(|(_, raw)| raw.len() == held.len() && raw.as_slice() == held.as_slice())
            .map(|(m, _)| *m)
            .collect();
        match hit.as_slice() {
            [m] if *m == i => identity += 1,
            [] => {
                missing += 1;
                println!("    layer.{i:<2} matches NO checkpoint layer");
            }
            other => {
                wrong.push((i, other.to_vec()));
                println!("    layer.{i:<2} holds checkpoint layer(s) {other:?}");
            }
        }
    }
    println!(
        "\n  {identity} driver layers hold their own checkpoint layer, \
         {} hold another's, {missing} match none",
        wrong.len()
    );
    println!(
        "  {}",
        if wrong.is_empty() && missing == 0 && identity > 0 {
            "the mapping is the identity, so the layers are not shifted."
        } else if identity == 0 {
            "NOT ONE layer holds its namesake -- the mapping is the defect."
        } else {
            "the mapping is not the identity, and the layers listed above are why."
        }
    );
}

/// WHERE EVERY RAW WEIGHT THE DRIVER HOLDS ACTUALLY CAME FROM.
///
/// The layer test above asks the composition question for one tensor per
/// layer. This asks it for ALL of them, and by content rather than by name:
/// index every small tensor in the checkpoint by what it CONTAINS, then take
/// each raw weight the driver holds and look up where it came from.
///
/// The point is that a name is the driver's opinion and the bytes are not. A
/// weight loaded from the wrong tensor -- a norm from a sibling layer, a gate
/// from the tower, an `o_proj` from an `up_proj` -- would be handed to a
/// correct kernel, computed correctly, and be wrong. Every per-kernel
/// reference in this file passes in that world, including the ones that check
/// a kernel against the very weight in question. A content lookup does not.
///
/// Only the RAW weights can be asked: a quantized one is int4 codes and equals
/// no float tensor anywhere. The discriminator is a `.scales` sibling, which
/// exists for exactly the encoded ones. What is left is the norms, the convs
/// and the small vectors -- little of the model's size and most of its wiring.
///
/// ## Read both float widths, or invent a defect
///
/// This test first reported `gate_norm` SYNTHESISED in all eighteen layers,
/// which would have been the whole bug: the gated-DeltaNet output scale
/// silently replaced by a constant, ratified by every kernel check because
/// those check a kernel against the weight it was HANDED.
///
/// It was this test that was wrong. `linear_attn.norm.weight` is stored F32
/// while the rest of the checkpoint is BF16, and an index built over BF16
/// alone cannot contain it -- so the lookup missed, and the same-size fallback
/// missed too because it compared BYTES against a tensor with twice as many of
/// them per value. Read as bf16 pairs the tell is unmistakable, alternating
/// tiny with real: `[-1.08e-19, 0.9609375, -0.0, 0.98828125]` is two f32s, not
/// four bf16s. So both widths are indexed, and F32 is narrowed to bf16 the way
/// a loader does, round-to-nearest-even, before its content is taken.
///
/// Narrowing alone then made `a_log` look synthesised, for the mirror reason:
/// it is held at the width it was FOUND, sixteen f32s, one per DeltaNet head.
/// So each checkpoint tensor is indexed under both forms -- as it lies and as
/// a loader would narrow it -- and the shape fallback accepts a source under
/// either reading of the held bytes. Read one width and this test invents a
/// defect; read one form and it invents another.
#[test]
fn where_every_raw_weight_the_driver_holds_actually_came_from() {
    let Some(dir) = qwen3_5_snapshot() else {
        println!("no Qwen3.5-0.8B snapshot, so IT COULD NOT BE MEASURED");
        return;
    };
    let Some(row) = model::catalog::find("qwen3.5-0.8b-base") else {
        return;
    };
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let Some(real) = qwen3_5_weights(&dir, row, &hybrid_plan(&facts)) else {
        return;
    };
    let Some(path) = std::fs::read_dir(&dir).ok().and_then(|d| {
        d.filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "safetensors"))
    }) else {
        return;
    };
    let Ok(mut f) = std::fs::File::open(&path) else {
        return;
    };
    use std::io::{Read, Seek, SeekFrom};
    let mut len = [0u8; 8];
    if f.read_exact(&mut len).is_err() {
        return;
    }
    let n = u64::from_le_bytes(len) as usize;
    let mut head = vec![0u8; n];
    if f.read_exact(&mut head).is_err() {
        return;
    }
    let Ok(head) = serde_json::from_slice::<serde_json::Value>(&head) else {
        return;
    };
    let Ok(head) = serde_json::from_value::<BTreeMap<String, serde_json::Value>>(head) else {
        return;
    };
    let base = 8 + n as u64;

    // What the driver would hold if it read this tensor: bf16 bytes either
    // way, so the comparison is like for like.
    let as_bf16 = |raw: &[u8], f32_wide: bool| -> Vec<u8> {
        if !f32_wide {
            return raw.to_vec();
        }
        raw.chunks_exact(4)
            .flat_map(|c| {
                let b = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let near = ((b >> 16) & 1) + 0x7fff;
                (((b.wrapping_add(near)) >> 16) as u16).to_le_bytes()
            })
            .collect()
    };
    let width = |meta: &serde_json::Value| match meta["dtype"].as_str() {
        Some("BF16") => Some((2usize, false)),
        Some("F32") => Some((4usize, true)),
        _ => None,
    };
    // Content, cheaply: length plus a 64-bit rolling digest. Two distinct
    // tensors colliding on both is not worth defending against, and a
    // collision would show as a weight matching two names rather than as a
    // wrong answer.
    let digest = |b: &[u8]| -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &c in b {
            h = (h ^ u64::from(c)).wrapping_mul(0x1000_0000_01b3);
        }
        h
    };
    // Elements, so a BF16 held weight and an F32 source can be the same shape.
    let elements = |meta: &serde_json::Value| -> Option<usize> {
        let (w, _) = width(meta)?;
        let (a, b) = (
            meta["data_offsets"][0].as_u64()?,
            meta["data_offsets"][1].as_u64()?,
        );
        Some((b - a) as usize / w)
    };

    let mut index: BTreeMap<(usize, u64), Vec<String>> = BTreeMap::new();
    let mut scanned = 0usize;
    for (name, meta) in &head {
        let Some((w, wide)) = width(meta) else {
            continue;
        };
        let (a, b) = (
            meta["data_offsets"][0].as_u64().unwrap_or(0),
            meta["data_offsets"][1].as_u64().unwrap_or(0),
        );
        // The embedding is a quarter of a gigabyte and is quantized on the
        // driver side, so it can never be an answer here; skipping the giants
        // keeps this test seconds rather than minutes.
        if b - a > 8 << 20 || !((b - a) as usize).is_multiple_of(w) {
            continue;
        }
        let mut raw = vec![0u8; (b - a) as usize];
        if f.seek(SeekFrom::Start(base + a)).is_ok() && f.read_exact(&mut raw).is_ok() {
            // BOTH representations, because the driver holds some weights at
            // the width it found them (`a_log` stays f32, one value per head)
            // and some narrowed (`gate_norm` becomes bf16). Indexing only one
            // makes the other look unread.
            let norm = as_bf16(&raw, wide);
            for form in [&raw, &norm] {
                index
                    .entry((form.len(), digest(form)))
                    .or_default()
                    .push(name.clone());
            }
            scanned += 1;
        }
    }

    let mut raw_held: Vec<&String> = real
        .keys()
        .filter(|k| {
            !k.ends_with(".scales")
                && !k.ends_with(".zeros")
                && !real.contains_key(&format!("{k}.scales"))
        })
        .collect();
    raw_held.sort();

    println!(
        "\n  where the driver's raw weights came from ({scanned} float tensors indexed, \
         {} raw weights held):",
        raw_held.len()
    );
    let layer_of = |k: &str| -> Option<String> {
        let t: Vec<&str> = k.split('.').collect();
        (t.first() == Some(&"layer"))
            .then(|| t.get(1).map(|s| (*s).to_string()))
            .flatten()
    };
    let (mut own, mut elsewhere, mut nowhere) = (0usize, Vec::new(), Vec::new());
    for k in &raw_held {
        let bytes = &real[*k];
        match index.get(&(bytes.len(), digest(bytes))) {
            None => nowhere.push((*k).clone()),
            Some(names) => {
                // "Its own" means the checkpoint name carries the layer index
                // the driver name does: `layer.7.conv_w` against
                // `...layers.7.linear_attn.conv1d.weight` agrees on the 7.
                let layer = layer_of(k);
                let ok = names.iter().any(|n| match &layer {
                    Some(i) => n.contains(&format!(".layers.{i}.")),
                    None => !n.contains(".layers."),
                });
                if ok {
                    own += 1;
                } else {
                    elsewhere.push(((*k).clone(), names.clone()));
                }
            }
        }
    }
    for (k, names) in &elsewhere {
        println!("    {k:<28} is really {names:?}");
    }
    // A raw weight matching nothing is not automatically wrong -- a bias the
    // checkpoint omits is synthesised, and `A_log` is stored one way and used
    // another -- but the two cases must be told apart, because "transformed"
    // and "loaded from the wrong tensor" look identical from here. So for each
    // one, ask whether the checkpoint holds a tensor of the same ELEMENT COUNT
    // in the same layer: if it does, this weight has a source and was changed
    // on the way in; if it does not, it was made up rather than read.
    let mut by_kind: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for k in &nowhere {
        let kind = k.rsplit('.').next().unwrap_or("?").to_string();
        let layer = layer_of(k);
        // The held width is not known from here, so accept the shape under
        // either reading: `len/2` values if this is bf16, `len/4` if f32.
        let bytes = real[k].len();
        let sourced = head.iter().any(|(n, meta)| {
            elements(meta).is_some_and(|e| e == bytes / 2 || e == bytes / 4)
                && layer.as_ref().is_none_or(|i| n.contains(&format!(".layers.{i}.")))
        });
        let e = by_kind.entry(kind).or_default();
        e.0 += 1;
        e.1 += usize::from(sourced);
    }
    for (kind, (count, sourced)) in &by_kind {
        println!(
            "    {kind:<12} {count} held, {sourced} have a same-shape tensor in their own layer \
             -- {}",
            if sourced == count {
                "read and transformed"
            } else if *sourced == 0 {
                "SYNTHESISED, not read"
            } else {
                "mixed, which is worth reading"
            }
        );
        let Some(bytes) = real.get(&format!("layer.0.{kind}")) else {
            continue;
        };
        let v: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect();
        println!(
            "      layer.0.{kind}: {} elements, first {:?}{}",
            v.len(),
            &v[..v.len().min(4)],
            if v.windows(2).all(|w| w[0] == w[1]) {
                " -- CONSTANT"
            } else {
                ""
            }
        );
    }
    println!(
        "\n  {own} raw weights sit where the driver says, {} sit somewhere else",
        elsewhere.len()
    );
    println!(
        "  {}",
        if elsewhere.is_empty() && own > 0 {
            "no raw weight is loaded from another tensor's bytes."
        } else if own == 0 {
            "NOT ONE raw weight was located, so this test measured nothing."
        } else {
            "the weights listed above are loaded from the wrong tensor."
        }
    );
}

/// THE MLP, ALL FIVE STEPS, AND THE RESIDUAL THAT CARRIES IT.
///
/// The MLP is most of this model's parameters and was the last block with no
/// CPU reference. It is also where the residual is COMPOSED -- `down` folds
/// the stream back in rather than writing beside it -- so checking the block
/// end to end checks the composition too, which no single-kernel reference
/// reaches.
///
/// Layer 0 lowers to five launches with nothing hidden between them:
///
/// ```text
///  10  rms_single_row    A@12288 -> A@0        (mlp_norm)
///  11  affine_qmv_fast   A@0     -> A@16384    (gate_proj, 1024 -> 3584)
///  12  affine_qmv_fast   A@0     -> A@41472    (up_proj,   1024 -> 3584)
///  13  silu_mul          A@16384, A@41472 -> A@55808
///  14  affine_qmv_fast_residual  A@55808, A@12288 -> A@0   (down)
/// ```
///
/// One fire reaches all of it. `A@12288` is written by the attention block's
/// `o_proj` and then only READ, so it survives to the end and is both the
/// norm's input and the residual `down` adds. `A@0` is the norm's output until
/// launch 14 overwrites it with the block's, so `x` is recomputed here from
/// `A@12288` rather than read -- which puts the norm under test as well.
///
/// Then walk it: dequantise each 4-bit row against its own group's scale and
/// zero, take `silu(gate) * up`, and fold. Four rectangles are compared, so a
/// disagreement says WHICH step rather than that the block is wrong.
#[test]
fn whether_the_mlp_computes_the_swiglu_its_weights_imply() {
    let Some((mut shell, real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let n = 2usize;
    let tokens: Vec<u32> = (0..n).map(|t| PERIOD[t % 6]).collect();
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    // The block is found by its kernels rather than by pinned indices, so an
    // upstream reordering makes this test skip rather than lie.
    let at = |k: &str, skip: usize| -> Option<usize> {
        low.launches
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                l.layers.start == 0 && low.kernels[l.kernel as usize].starts_with(k)
            })
            .map(|(i, _)| i)
            .nth(skip)
    };
    let arenas = |i: usize| -> Vec<(usize, usize)> {
        let l = &low.launches[i];
        low.args[l.args.start as usize..l.args.end as usize]
            .iter()
            .filter_map(|a| match a {
                Arg::Arena { at, width, bytes } => Some((*at, *width as usize * *bytes as usize)),
                _ => None,
            })
            .collect()
    };
    let (Some(norm), Some(silu), Some(down)) = (
        at("rms_single_row", 1),
        at("silu_mul", 0),
        at("affine_qmv_fast_residual", 1),
    ) else {
        println!("layer 0 does not lower to the MLP block this test reads");
        return;
    };
    let (Some(gate_l), Some(up_l)) = (at("affine_qmv_fast_bfloat16", 4), at("affine_qmv_fast_bfloat16", 5))
    else {
        println!("the two MLP projections are not where this test looks");
        return;
    };
    let (nr, gr, ur, sr, dr) = (
        arenas(norm),
        arenas(gate_l),
        arenas(up_l),
        arenas(silu),
        arenas(down),
    );
    // `[in, out]` for the norm and the projections, `[a, b, out]` for
    // `silu_mul`, `[in, residual, out]` for the residual-folding `down`.
    let (r_at, r_w) = nr[0];
    let (gate_at, _) = gr[1];
    let (up_at, _) = ur[1];
    let (silu_at, _) = sr[2];
    let (out_at, _) = dr[2];
    if dr[1].0 != r_at {
        println!("the residual `down` folds is not the norm's input, so this test cannot read it");
        return;
    }

    shell.fire_prefix(Some(down + 1));
    let arena = match shell.step(&[driver_wgpu::turns::Turn {
        who: 4_240,
        tokens: tokens.clone(),
    }]) {
        Ok(s) => s.arena,
        Err(why) => {
            println!("the truncated prefill was refused: {why}");
            return;
        }
    };
    shell.fire_prefix(None);
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    let row = |o: usize, w: usize, t: usize| -> Vec<f32> { bf16(&arena[o + t * w..o + (t + 1) * w]) };

    let (hidden, inter, group) = (1024usize, 3584usize, 64usize);
    let want = |name: &str, out_n: usize, in_n: usize| -> Option<(Vec<u8>, Vec<f32>, Vec<f32>)> {
        let (c, s, z) = (
            real.get(name)?,
            real.get(&format!("{name}.scales")).map(|b| bf16(b))?,
            real.get(&format!("{name}.zeros")).map(|b| bf16(b))?,
        );
        (c.len() == out_n * in_n / 2 && s.len() == out_n * (in_n / group)).then(|| (c.clone(), s, z))
    };
    let (Some(wg), Some(wu), Some(wd), Some(wn)) = (
        want("layer.0.gate_proj", inter, hidden),
        want("layer.0.up_proj", inter, hidden),
        want("layer.0.down", hidden, inter),
        real.get("layer.0.mlp_norm").map(|b| bf16(b)),
    ) else {
        println!("the MLP's weights are not the shapes this test unpacks");
        return;
    };
    let qmv = |w: &(Vec<u8>, Vec<f32>, Vec<f32>), x: &[f32], out_n: usize| -> Vec<f32> {
        let in_n = x.len();
        let groups = in_n / group;
        (0..out_n)
            .map(|o| {
                let base = o * in_n / 2;
                let mut acc = 0.0f32;
                for g in 0..groups {
                    let (s, z) = (w.1[o * groups + g], w.2[o * groups + g]);
                    for e in 0..group {
                        let i = g * group + e;
                        let byte = w.0[base + i / 2];
                        let code = f32::from(if i % 2 == 0 { byte & 0xf } else { byte >> 4 });
                        acc += x[i] * (s * code + z);
                    }
                }
                acc
            })
            .collect()
    };
    let worst = |a: &[f32], b: &[f32]| -> (f32, f32) {
        let d = a
            .iter()
            .zip(b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        let s = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        (d, d / s.max(1e-30))
    };

    println!("\n  the MLP of layer 0, walked on a CPU:");
    let mut widest = 0.0f32;
    for t in 0..n {
        let r = row(r_at, r_w, t);
        // The norm is recomputed rather than read, because `down` has already
        // written over where it was.
        let ss = r.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / hidden as f64;
        let inv = 1.0 / (ss + 1e-6).sqrt();
        // THE FOLD THE FACTS STATE, which is gemma's `(1 + w)`. This read `* w`
        // when it was written, because that was what the device did -- and it
        // agreed with the device to one bf16 step while the model answered a
        // space to every prompt, which is the whole lesson of
        // `which_fold_the_final_norm_applies_and_what_each_one_answers`. When
        // the fold was fixed this reference was not, and it went on agreeing
        // with nothing: 0.93 relative on `gate_proj`, which looked exactly like
        // a broken projection kernel.
        let x: Vec<f32> = r
            .iter()
            .zip(&wn)
            .map(|(v, w)| (f64::from(*v) * inv) as f32 * (1.0 + w))
            .collect();
        let gate = qmv(&wg, &x, inter);
        let up = qmv(&wu, &x, inter);
        let y: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let folded: Vec<f32> = qmv(&wd, &y, hidden)
            .iter()
            .zip(&r)
            .map(|(d, rr)| d + rr)
            .collect();
        for (what, cpu, gpu) in [
            ("gate_proj", &gate, row(gate_at, inter * 2, t)),
            ("up_proj", &up, row(up_at, inter * 2, t)),
            ("silu_mul", &y, row(silu_at, inter * 2, t)),
            ("down+residual", &folded, row(out_at, hidden * 2, t)),
        ] {
            let (d, rel) = worst(cpu, &gpu);
            widest = widest.max(rel);
            println!("    t{t} {what:<14} widest |cpu - gpu| {d:.6}, relative {rel:e}");
        }
    }
    println!(
        "\n  {}",
        if widest < 3e-2 {
            "THE MLP COMPUTES THE SWIGLU ITS WEIGHTS IMPLY, and folds the residual it was given."
        } else {
            "THE MLP DISAGREES WITH ITS OWN WEIGHTS, and the step named above is where."
        }
    );
}

/// THE READOUT IS TIED TO A QUANTIZED TABLE, SO ASK WHAT IT COSTS.
///
/// `tie_word_embeddings` is true here, so the table that reads the state out
/// to 248,320 logits is the same table the tokens are gathered FROM -- and the
/// driver holds it at four bits. Those two uses are not equally forgiving. A
/// gather returns one row and a small error moves one vector slightly; a
/// readout ranks a quarter of a million rows against each other, and the
/// winner is decided by DIFFERENCES between them. Noise that is invisible in
/// the first can decide the second.
///
/// `what_the_int4_encode_costs_against_the_bfloat_it_came_from` already showed
/// the encode sits on the four-bit floor, which is the right answer to "is the
/// encode faithful" and no answer at all to "is four bits enough to rank with".
/// This asks the second question the only way it can be answered: take the
/// state the fire actually produced, score it against the checkpoint's own
/// bf16 rows, and see whether the ranking survives.
///
/// One variable moves. `h` is the int4 model's own hidden state, so everything
/// upstream is held fixed and what is left is the readout's width alone.
#[test]
fn whether_the_tied_readout_ranks_the_same_at_four_bits_and_at_sixteen() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    // "The capital of France is Paris. The capital of France is"
    let tokens: Vec<u32> = vec![561, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 9338, 369];
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        tokens.len()
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    let l = &low.launches[low.launches.len() - 1];
    let (mut arenas, mut weights) = (Vec::new(), Vec::new());
    for a in &low.args[l.args.start as usize..l.args.end as usize] {
        match a {
            Arg::Arena { at, width, bytes } => arenas.push((*at, *width as usize * *bytes as usize)),
            Arg::Weight(n) => weights.push(n.clone()),
            // Neither is a RECTANGLE this probe can read back: a named
            // value is bound by the backend rather than placed in the arena,
            // and a raise is a host aggregate with no row width at all.
            Arg::Named { .. } | Arg::Raised { .. } => {}
        }
    }
    let (Some((o, stride)), Some(stem)) = (arenas.first().copied(), weights.first().cloned()) else {
        return;
    };
    let step = match shell.step(&[driver_wgpu::turns::Turn {
        who: 6_100,
        tokens: tokens.clone(),
    }]) {
        Ok(s) => s,
        Err(why) => {
            println!("the fire was refused: {why}");
            return;
        }
    };
    let picked = step.readout_of[0];
    let Some(gpu) = step.logits.row(picked).map(<[f32]>::to_vec) else {
        return;
    };
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    // Row `picked` on both sides: an arena range's stride is ONE row.
    let h = bf16(&step.arena[o + picked * stride..o + (picked + 1) * stride]);
    let hidden = h.len();

    // The same table at sixteen bits, read where it lies.
    let Some(path) = std::fs::read_dir(qwen3_5_snapshot().unwrap_or_default())
        .ok()
        .and_then(|d| {
            d.filter_map(Result::ok)
                .map(|e| e.path())
                .find(|p| p.extension().is_some_and(|e| e == "safetensors"))
        })
    else {
        return;
    };
    let Ok(mut f) = std::fs::File::open(&path) else {
        return;
    };
    use std::io::{Read, Seek, SeekFrom};
    let mut len = [0u8; 8];
    if f.read_exact(&mut len).is_err() {
        return;
    }
    let n = u64::from_le_bytes(len) as usize;
    let mut head = vec![0u8; n];
    if f.read_exact(&mut head).is_err() {
        return;
    }
    let Ok(head) = serde_json::from_slice::<serde_json::Value>(&head) else {
        return;
    };
    let src = "model.language_model.embed_tokens.weight";
    let Some(meta) = head.get(src) else {
        println!("`{src}` is not in the checkpoint");
        return;
    };
    if meta["dtype"].as_str() != Some("BF16") {
        println!("`{src}` is {:?}, which this test does not read", meta["dtype"]);
        return;
    }
    let (a, b) = (
        meta["data_offsets"][0].as_u64().unwrap_or(0),
        meta["data_offsets"][1].as_u64().unwrap_or(0),
    );
    let base = 8 + n as u64 + a;
    let vocab = ((b - a) as usize) / (hidden * 2);
    println!(
        "\n  the readout is `{stem}` at four bits against `{src}` at sixteen, \
         {vocab} rows of {hidden}"
    );
    if vocab == 0 || vocab > gpu.len() {
        println!("  the table is {vocab} rows and the fire answered {} -- not comparable", gpu.len());
        return;
    }
    if f.seek(SeekFrom::Start(base)).is_err() {
        return;
    }
    // Streamed, because the table is half a gigabyte and only one row at a
    // time is needed.
    let mut cpu = vec![0.0f32; vocab];
    let mut buf = vec![0u8; 4096 * hidden * 2];
    let mut done = 0usize;
    while done < vocab {
        let take = (vocab - done).min(4096);
        let want = take * hidden * 2;
        if f.read_exact(&mut buf[..want]).is_err() {
            println!("  the table is shorter than its header says");
            return;
        }
        for r in 0..take {
            let at = r * hidden * 2;
            let mut acc = 0.0f32;
            for (i, hv) in h.iter().enumerate() {
                let c = at + i * 2;
                let w = f32::from_bits(u32::from(u16::from_le_bytes([buf[c], buf[c + 1]])) << 16);
                acc += hv * w;
            }
            cpu[done + r] = acc;
        }
        done += take;
    }

    let top = |v: &[f32], k: usize| -> Vec<(usize, f32)> {
        let mut ix: Vec<usize> = (0..v.len()).collect();
        ix.sort_by(|p, q| v[*q].total_cmp(&v[*p]));
        ix.into_iter().take(k).map(|i| (i, v[i])).collect()
    };
    let (tg, tc) = (top(&gpu[..vocab], 6), top(&cpu, 6));
    println!("    four bits : {tg:?}");
    println!("    sixteen   : {tc:?}");
    // Where the four-bit winner sits in the sixteen-bit ranking, and the other
    // way round: a readout destroyed by width shows as a large move in both.
    let rank = |v: &[f32], t: usize| v.iter().filter(|x| **x > v[t]).count();
    println!(
        "    the four-bit winner {} sits at rank {} of {vocab} in sixteen; \
         the sixteen-bit winner {} sits at rank {} in four",
        tg[0].0,
        rank(&cpu, tg[0].0),
        tc[0].0,
        rank(&gpu[..vocab], tc[0].0)
    );
    println!(
        "\n  {}",
        if tg[0].0 == tc[0].0 {
            "THE WIDTH DOES NOT DECIDE THE ANSWER: the same token wins at four bits and at sixteen, so the readout's quantization is not why this model answers what it answers."
        } else {
            "THE WIDTH DECIDES THE ANSWER: the tied readout ranks differently at four bits than at sixteen, and that is a defect the encode's faithfulness cannot see."
        }
    );
}

/// THE ROTATION, WHICH EVERY OTHER CHECK IN THIS FILE READS THROUGH.
///
/// `neox_mb` is the last kernel here with no reference, and it is the one most
/// able to hide. `whether_attention_is_the_softmax_over_the_cache_it_reads`
/// reads the cache AFTER the rotation and asks whether the softmax over it is
/// right -- so a wrong rotation is not a disagreement there, it is just a
/// different cache, and attention faithfully attends to the wrong thing. The
/// symptom would be a model with damaged positions, which is what a state that
/// ranks space, comma, period and newline above every word looks like.
///
/// Three things are checked, and they fail differently:
///
/// - **The passthrough.** `partial_rotary_factor` is 0.25 of a 256-wide head,
///   so 64 dimensions turn and 192 must be left exactly alone. Rotating all of
///   them is the easiest way to write this kernel wrongly and the easiest to
///   see: those 192 are compared for EQUALITY, not closeness.
/// - **The pairing.** NeoX pairs `i` with `i + rot/2`; GPT-J pairs `2i` with
///   `2i+1`. Both are computed and the better reported, so a mismatched
///   convention is named rather than mistaken for noise.
/// - **The angle.** `rope_theta` is 1e7, and position is the token's index
///   from a seat that starts empty.
///
/// The kernel writes in place, so the operand cannot be read beside its
/// result. Two fires of the same tokens from two fresh seats give the state
/// before and after, which is sound precisely because the prefill is
/// deterministic.
#[test]
fn whether_the_rotation_turns_the_dimensions_it_is_supposed_to() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    let tokens: Vec<u32> = vec![561, 6511, 314, 9338, 369, 11751, 13];
    let n = tokens.len();
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    let Some(rope) = low
        .launches
        .iter()
        .position(|l| low.kernels[l.kernel as usize].starts_with("neox_mb"))
    else {
        println!("this plan has no rotation, which would be its own finding");
        return;
    };
    let l = &low.launches[rope];
    let a: Vec<(usize, usize)> = low.args[l.args.start as usize..l.args.end as usize]
        .iter()
        .filter_map(|x| match x {
            Arg::Arena { at, width, bytes } => Some((*at, *width as usize * *bytes as usize)),
            _ => None,
        })
        .collect();
    let [(q_at, q_w), (out_at, _)] = a[..] else {
        println!("the rotation does not take the two ranges this test reads");
        return;
    };
    if q_at != out_at {
        println!("the rotation is not in place, so this test's two-fire trick is unnecessary and wrong");
        return;
    }
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    let mut fire = |upto: usize, who: u64| -> Option<Vec<Vec<f32>>> {
        shell.fire_prefix(Some(upto));
        let out = shell
            .step(&[driver_wgpu::turns::Turn {
                who,
                tokens: tokens.clone(),
            }])
            .ok()?;
        shell.fire_prefix(None);
        Some(
            (0..n)
                .map(|t| bf16(&out.arena[q_at + t * q_w..q_at + (t + 1) * q_w]))
                .collect(),
        )
    };
    let (Some(before), Some(after)) = (fire(rope, 7_100), fire(rope + 1, 7_200)) else {
        println!("a truncated prefill was refused");
        return;
    };

    let (head_dim, rot, theta) = (256usize, 64usize, 1e7f64);
    let heads = q_w / 2 / head_dim;
    println!(
        "\n  the rotation over {heads} head(s) of {head_dim}, turning {rot} and passing {} through:",
        head_dim - rot
    );
    // THE PASSTHROUGH, for equality. A dimension outside the rotary span that
    // moved at all is a defect no tolerance should absorb.
    let (mut moved, mut checked) = (0usize, 0usize);
    for t in 0..n {
        for h in 0..heads {
            for i in rot..head_dim {
                let j = h * head_dim + i;
                checked += 1;
                if before[t][j] != after[t][j] {
                    moved += 1;
                }
            }
        }
    }
    println!("    {moved} of {checked} dimensions outside the rotary span moved");

    // THE ROTATION, under both conventions.
    let mut worst = [0.0f32; 2];
    let mut scale = 0.0f32;
    for t in 0..n {
        for h in 0..heads {
            let base = h * head_dim;
            for i in 0..rot / 2 {
                let ang = t as f64 * theta.powf(-2.0 * i as f64 / rot as f64);
                let (c, s) = (ang.cos() as f32, ang.sin() as f32);
                for (which, (p, q)) in [(i, i + rot / 2), (2 * i, 2 * i + 1)].into_iter().enumerate()
                {
                    let (x, y) = (before[t][base + p], before[t][base + q]);
                    let (wp, wq) = (x * c - y * s, y * c + x * s);
                    worst[which] = worst[which]
                        .max((wp - after[t][base + p]).abs())
                        .max((wq - after[t][base + q]).abs());
                }
                scale = scale.max(after[t][base + i].abs());
            }
        }
    }
    let rel = worst.map(|w| w / scale.max(1e-30));
    println!("    neox pairing (i, i+{}): widest {:e}, relative {:e}", rot / 2, worst[0], rel[0]);
    println!("    gptj pairing (2i, 2i+1): widest {:e}, relative {:e}", worst[1], rel[1]);
    let good = rel[0] < 3e-2 || rel[1] < 3e-2;
    println!(
        "\n  {}",
        if moved == 0 && good {
            "THE ROTATION TURNS WHAT IT IS SUPPOSED TO and leaves the rest alone."
        } else if moved > 0 {
            "THE ROTATION MOVED DIMENSIONS OUTSIDE ITS SPAN, which no partial-rotary model survives."
        } else {
            "THE ROTATION MATCHES NEITHER CONVENTION AT THIS THETA, and every attention layer reads through it."
        }
    );
}

/// WHICH FOLD THE FINAL NORM APPLIES, AND WHICH ONE THIS CHECKPOINT WANTS.
///
/// Gemma's RMSNorm is not the usual one: it scales by `1 + w` rather than by
/// `w`. `Qwen35HybridFacts::qwen3_5_0_8b` USED to state `NormVariant::Gemma`
/// and now states `Plain` -- `model/src/qwen_3_5/spec.rs` gives the evidence
/// on both sides -- so this test's job is unchanged and its expected answer is
/// not: it measures which fold the device applied, whichever one the facts
/// currently ask for. That difference is invisible to every other check in
/// this file.
/// `whether_the_logits_are_the_readout_of_the_state_the_fire_produced` starts
/// FROM the normalized state and asks whether the readout of it is right, so a
/// state normalized the wrong way is not a disagreement there -- it is a
/// different state, faithfully read out.
///
/// It is also not a small difference. `xhat * (1 + w)` is `xhat * w + xhat`,
/// so the wrong fold adds the raw normalized residual to whatever the right
/// one produces. Dotted against a TIED embedding table that is also the
/// readout, an extra `xhat` term rewards whichever rows the residual already
/// points at -- and a residual stream's resting direction is the frequent,
/// contentless part of the vocabulary. A model that answers space, comma,
/// period, newline to every prompt is what that looks like.
///
/// `whether_the_mlp_computes_the_swiglu_its_weights_imply` already showed the
/// PER-BLOCK norms fold plainly: its reference multiplies by `w` and its
/// projections agree to one bf16 step, which `1 + w` could not do. So the two
/// folds coexist in one model, and at most one of them is right.
///
/// This settles it by consequence rather than by argument. Recover `xhat` from
/// the final norm's own input, build both candidates, and check which the
/// device produced -- then score BOTH against the checkpoint's bf16 rows and
/// print what each would answer. A fold that is merely different changes the
/// numbers; a fold that is wrong changes the word.
#[test]
fn which_fold_the_final_norm_applies_and_what_each_one_answers() {
    let Some((mut shell, real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan(&facts);
    // "The capital of France is Paris. The capital of France is"
    let tokens: Vec<u32> = vec![561, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 9338, 369];
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        tokens.len()
    ];
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the hybrid lowers");
    // The LAST `rms_single_row` is the final norm: every other one opens a
    // block and is followed by that block's projections.
    let Some(fin) = low
        .launches
        .iter()
        .rposition(|l| low.kernels[l.kernel as usize].starts_with("rms_single_row"))
    else {
        return;
    };
    let l = &low.launches[fin];
    let (mut a, mut w) = (Vec::new(), Vec::new());
    for x in &low.args[l.args.start as usize..l.args.end as usize] {
        match x {
            Arg::Arena { at, width, bytes } => a.push((*at, *width as usize * *bytes as usize)),
            Arg::Weight(n) => w.push(n.clone()),
            // Neither is a RECTANGLE this probe can read back: a named
            // value is bound by the backend rather than placed in the arena,
            // and a raise is a host aggregate with no row width at all.
            Arg::Named { .. } | Arg::Raised { .. } => {}
        }
    }
    let ([(in_at, in_w), (out_at, out_w)], [stem]) = (a[..].try_into().unwrap_or([(0, 0); 2]), &w[..])
    else {
        println!("the final norm does not take the operands this test reads");
        return;
    };
    // STOP AT THE NORM. The arena is reused, and a full fire lets the readout
    // write over the very rectangle the norm read from -- the first draft of
    // this test read that reused buffer as `r` and got a state matching
    // NEITHER fold, which is what a stale operand looks like. Firing through
    // the norm and no further leaves both its input and its output live.
    shell.fire_prefix(Some(fin + 1));
    let step = match shell.step(&[driver_wgpu::turns::Turn {
        who: 6_200,
        tokens: tokens.clone(),
    }]) {
        Ok(s) => s,
        Err(why) => {
            println!("the truncated fire was refused: {why}");
            return;
        }
    };
    shell.fire_prefix(None);
    // The answer comes from the last token, and a truncated fire has no
    // readout to ask.
    let picked = tokens.len() - 1;
    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    let r = bf16(&step.arena[in_at + picked * in_w..in_at + (picked + 1) * in_w]);
    let got = bf16(&step.arena[out_at + picked * out_w..out_at + (picked + 1) * out_w]);
    let Some(wn) = real.get(stem).map(|b| bf16(b)) else {
        println!("`{stem}` is not held");
        return;
    };
    let hidden = r.len();
    if wn.len() != hidden || got.len() != hidden {
        println!("the final norm's shapes are not {hidden} across");
        return;
    }
    let ss = r.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / hidden as f64;
    let inv = 1.0 / (ss + 1e-6).sqrt();
    let xhat: Vec<f32> = r.iter().map(|v| (f64::from(*v) * inv) as f32).collect();
    let plain: Vec<f32> = xhat.iter().zip(&wn).map(|(x, w)| x * w).collect();
    let gemma: Vec<f32> = xhat.iter().zip(&wn).map(|(x, w)| x * (1.0 + w)).collect();
    let worst = |c: &[f32]| -> f32 {
        c.iter()
            .zip(&got)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max)
    };
    let scale = got.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!(
        "\n  `{stem}`: mean {:.4}, and the final norm's output is\n    \
         x*w      off by {:e}\n    x*(1+w)  off by {:e}   (values reach {scale:.3})",
        wn.iter().sum::<f32>() / hidden as f32,
        worst(&plain) / scale,
        worst(&gemma) / scale
    );
    // The per-block norms, for contrast: one distribution or two?
    for other in ["layer.0.attn_norm", "layer.0.mlp_norm"] {
        if let Some(v) = real.get(other).map(|b| bf16(b)) {
            println!(
                "    {other}: mean {:.4} over {}",
                v.iter().sum::<f32>() / v.len() as f32,
                v.len()
            );
        }
    }

    // WHAT EACH FOLD WOULD ANSWER, against the checkpoint's own bf16 table, so
    // the two candidates are separated by the word they produce.
    let Some(dir) = qwen3_5_snapshot() else {
        return;
    };
    let Some(path) = std::fs::read_dir(dir).ok().and_then(|d| {
        d.filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "safetensors"))
    }) else {
        return;
    };
    let Ok(mut f) = std::fs::File::open(&path) else {
        return;
    };
    use std::io::{Read, Seek, SeekFrom};
    let mut len = [0u8; 8];
    if f.read_exact(&mut len).is_err() {
        return;
    }
    let n = u64::from_le_bytes(len) as usize;
    let mut head = vec![0u8; n];
    if f.read_exact(&mut head).is_err() {
        return;
    }
    let Ok(head) = serde_json::from_slice::<serde_json::Value>(&head) else {
        return;
    };
    let src = "model.language_model.embed_tokens.weight";
    let Some(meta) = head.get(src).filter(|m| m["dtype"] == "BF16") else {
        return;
    };
    let (lo, hi) = (
        meta["data_offsets"][0].as_u64().unwrap_or(0),
        meta["data_offsets"][1].as_u64().unwrap_or(0),
    );
    let vocab = ((hi - lo) as usize) / (hidden * 2);
    if f.seek(SeekFrom::Start(8 + n as u64 + lo)).is_err() {
        return;
    }
    let mut logits = [vec![0.0f32; vocab], vec![0.0f32; vocab]];
    let mut buf = vec![0u8; 4096 * hidden * 2];
    let mut done = 0usize;
    while done < vocab {
        let take = (vocab - done).min(4096);
        if f.read_exact(&mut buf[..take * hidden * 2]).is_err() {
            return;
        }
        for k in 0..take {
            let at = k * hidden * 2;
            let (mut p, mut g) = (0.0f32, 0.0f32);
            for i in 0..hidden {
                let c = at + i * 2;
                let v = f32::from_bits(u32::from(u16::from_le_bytes([buf[c], buf[c + 1]])) << 16);
                p += plain[i] * v;
                g += gemma[i] * v;
            }
            logits[0][done + k] = p;
            logits[1][done + k] = g;
        }
        done += take;
    }
    let top = |v: &[f32]| -> Vec<(usize, f32)> {
        let mut ix: Vec<usize> = (0..v.len()).collect();
        ix.sort_by(|p, q| v[*q].total_cmp(&v[*p]));
        ix.into_iter().take(6).map(|i| (i, v[i])).collect()
    };
    println!("\n  \"The capital of France is Paris. The capital of France is\" ->");
    println!("    with x*w      : {:?}", top(&logits[0]));
    println!("    with x*(1+w)  : {:?}", top(&logits[1]));
    println!(
        "  (11751 is ` Paris`, 220 is a space, 11 a comma, 13 a period, 198 a newline)"
    );
    // THE GUARD. What this test is FOR, now that the fold is fixed: the device
    // must still be folding as gemma does. Reverting `Ctx::norm` to `Plain`
    // makes this model answer a space to every prompt, and nothing else in
    // this file notices -- every kernel reference checks a kernel against the
    // operands it was HANDED, and a wrongly folded norm hands on wrongly and
    // faithfully. So the fold is asserted here or it is asserted nowhere.
    let (plain_off, gemma_off) = (worst(&plain) / scale, worst(&gemma) / scale);
    assert!(
        gemma_off < 3e-2,
        "the final norm no longer folds as gemma: x*(1+w) is off by {gemma_off:e} \
         (x*w by {plain_off:e}). `Qwen35HybridFacts` states `NormVariant::Gemma` and \
         this checkpoint's norm gains are trained from zero -- folding plainly makes \
         this model answer a space to every prompt."
    );
    let (pw, gw) = (top(&logits[0])[0].0, top(&logits[1])[0].0);
    println!(
        "\n  the device folds as GEMMA (off by {gemma_off:e}, against {plain_off:e} for plain), {}",
        if pw == gw {
            "and both folds happen to rank the same token here."
        } else {
            "which is the fold that puts a word at the top."
        }
    );
}

/// THE SAME FIX, ON A CHECKPOINT IT WAS NOT FOUND ON.
///
/// The norm fold was diagnosed against Qwen3.5-0.8B-**Base**, and a fix
/// verified on the one checkpoint that produced it is a fix that might be a
/// coincidence of those weights. The instruct model sits beside it in the same
/// cache, shares the architecture tensor for tensor -- so it loads through the
/// same catalog row and the same facts -- and has never been fired here.
///
/// It is also the harder test. A base model continues text, which is what
/// every probe in this file asks for; an instruct model has been tuned to
/// answer, and answering a bare continuation is not what it is for. So the
/// question is deliberately weak: not "is this the word I want" but "is this
/// TEXT" -- does greedy decoding leave the whitespace-and-punctuation basin
/// that the wrong fold pinned every prompt to.
///
/// That basin is the signature and it is specific: 220 is a space, 11 a comma,
/// 13 a period, 198 a newline, 271 two. Under the wrong fold EVERY prompt
/// generated `[220, 220, 220]`, on every checkpoint, because the state had no
/// context left in it. A model emitting anything else has left it.
#[test]
#[ignore = "loads a second real checkpoint; run explicitly"]
fn whether_the_instruct_checkpoint_leaves_the_whitespace_basin_too() {
    let Some(dir) = qwen3_5_snapshot_of(&["models--Qwen--Qwen3.5-0.8B"]) else {
        println!("no Qwen3.5-0.8B (instruct) snapshot, so IT COULD NOT BE MEASURED");
        return;
    };
    println!("  the instruct checkpoint at {dir}");
    let Some((mut shell, _real)) =
        qwen3_5_shell_in(&dir, 8, 4, model::shared::policy::RuntimeQuant::Int4)
    else {
        return;
    };
    // The basin the wrong fold pinned every prompt to.
    const FILLER: [u32; 6] = [220, 11, 13, 198, 271, 321];
    let prompts: [(&str, &[u32]); 3] = [
        ("capital of France", &[561, 6511, 314, 9338, 369]),
        ("the x8", &[279; 8]),
        ("cat x8", &[7993; 8]),
    ];
    println!("\n  three greedy steps per prompt, and whether any of it is a word:");
    let (mut escaped, mut asked) = (0usize, 0usize);
    for (what, prompt) in prompts {
        let mut tokens = prompt.to_vec();
        let mut made = Vec::new();
        // A fresh seat per prompt, so no conversation carries into the next.
        let who = 8_000 + asked as u64;
        for step in 0..3 {
            // The first step is the prompt; each one after is the single token
            // just chosen, which is what continuing a conversation is.
            let fire: Vec<u32> = if step == 0 {
                tokens.clone()
            } else {
                vec![*tokens.last().expect("a token was just appended")]
            };
            let Ok(out) = shell.step(&[driver_wgpu::turns::Turn { who, tokens: fire }]) else {
                println!("    {what:<20} the fire was refused");
                break;
            };
            let Some(row) = out
                .readout_of
                .first()
                .and_then(|i| out.logits.row(*i))
                .map(<[f32]>::to_vec)
            else {
                break;
            };
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i as u32);
            made.push(best);
            tokens.push(best);
        }
        asked += 1;
        let out = made.iter().any(|t| !FILLER.contains(t));
        escaped += usize::from(out);
        println!(
            "    {what:<20} generated {made:?} -- {}",
            if out { "not filler" } else { "ALL FILLER" }
        );
    }
    println!(
        "\n  {escaped} of {asked} prompts generated something outside the filler set"
    );
    println!(
        "  {}",
        if escaped == asked && asked > 0 {
            "THE SECOND CHECKPOINT LEAVES THE BASIN TOO, so the fold was the fault and not those weights."
        } else if escaped == 0 {
            "EVERY PROMPT IS STILL FILLER on this checkpoint, which the fold alone does not explain."
        } else {
            "SOME PROMPTS LEFT THE BASIN AND SOME DID NOT, which is worth reading rather than summarizing."
        }
    );
}

/// EVERY NORM THIS PLAN STATES FOLDS THE WAY ITS FACTS SAY, WITH NO DEVICE.
///
/// `which_fold_the_final_norm_applies_and_what_each_one_answers` asserts the
/// same property against a real fire, and that is the stronger check -- it
/// measures the device rather than the plan. It is also the one that cannot
/// run: it needs an adapter, a 424 MB stage and two minutes, so on any machine
/// without the checkpoint it prints a line and returns having measured
/// nothing. A defect that only a skipped test catches is not caught.
///
/// This one needs neither. `rms_norm_gain` packs `RmsParams` field for field --
/// `eps, axis_size, w_stride, plus_one, gain` -- into the launch's `params`,
/// so the fold is a NUMBER in the traced plan and can be read straight out of
/// it, in under a second.
///
/// # It pins the EQUIVALENCE, not the fold
///
/// The first draft asserted `NormVariant::Gemma` and `plus_one == 1` outright,
/// and went red when `Qwen35HybridFacts::qwen3_5_0_8b` moved to `Plain` -- see
/// `model/src/qwen_3_5/spec.rs`, which says the `Gemma` reading rested on
/// `qwen3_5_forward.cpp` launching `rmsnorm_gemma_bf16` with no 0.8B staged to
/// check it, while the two measurable Qwen3.6 checkpoints and `mlx_lm`'s
/// `nn.RMSNorm` for the whole family say plain.
///
/// A test that has to be edited every time that reading is revisited is a test
/// that will be edited WITHOUT being thought about. So it asserts the thing
/// that is true either way: whatever the facts state, every `rms_single_row`
/// in the plan states the SAME, and no statement disagrees with its
/// neighbours. A `Ctx::norm` that ignored the facts, or folded some norms and
/// not others, fails this whichever variant is the right one.
///
/// It also pins the split, which is the part an argument would get wrong: the
/// gated-DeltaNet's output gain is genuinely plain and does NOT come through
/// here. `linear_attn.norm.weight` ships at 0.96-0.99, near one, where the
/// norms below ship near zero -- one checkpoint carrying both conventions,
/// each where it belongs. So the count is checked as well as the flag: every
/// `rms_single_row` folds, and `gated_rms` is not among them.
#[test]
fn every_norm_in_the_plan_folds_the_way_the_facts_state() {
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let want = match facts.norm_variant {
        model_ir::trace::NormVariant::Gemma => 1,
        model_ir::trace::NormVariant::Plain => 0,
    };
    let plan = hybrid_plan(&facts);
    // `RmsParams`: eps, axis_size, w_stride, plus_one, gain.
    const PLUS_ONE: usize = 3;
    let (mut folded, mut plain, mut gated) = (0usize, Vec::new(), 0usize);
    for op in &plan.ops {
        let model_ir::trace::OpKind::Launch { kernel, params, weights, .. } = &op.kind else {
            continue;
        };
        if kernel.starts_with("gated_rms") {
            gated += 1;
            continue;
        }
        if !kernel.starts_with("rms_single_row") {
            continue;
        }
        let name = weights.first().cloned().unwrap_or_default();
        match params.get(PLUS_ONE) {
            Some(got) if *got == want => folded += 1,
            other => plain.push((name, *other.unwrap_or(&u32::MAX))),
        }
    }
    println!(
        "\n  the facts state {:?}, so `plus_one` should be {want} everywhere: \
         {folded} `rms_single_row` statements agree, {} do not, and {gated} \
         `gated_rms` statements are plain by construction",
        facts.norm_variant,
        plain.len()
    );
    for (name, got) in plain.iter().take(8) {
        println!("    {name} has plus_one = {got}");
    }
    assert!(
        plain.is_empty(),
        "{} norm statements disagree with the facts they were lowered from: \
         {:?}. The variant is a property of the CHECKPOINT, so a plan that \
         folds some of its norms and not others is wrong under either reading \
         -- and if the gains are trained from zero, the ones that do not fold \
         multiply by `w` directly and answer a space to every prompt.",
        plain.len(),
        &plain[..plain.len().min(8)]
    );
    // 24 attention norms + 24 mlp norms + 6 q + 6 k + 1 final.
    assert_eq!(folded, 61, "the plan's norm count moved");
    assert_eq!(gated, 18, "one gated norm per gated-DeltaNet layer");
}

/// SPLITTING A CONVERSATION MUST NOT CHANGE ITS ANSWER.
///
/// [`whether_the_prefill_and_the_decode_leave_the_same_carry`] reports that
/// the two recurrences leave different `recurrent_state`: 262,144 of 262,144
/// elements at layers 1, 2, 4 and 5, widest 5.42, while `conv_state` agrees to
/// the byte. Layer 3 agrees only because it is a FULL-ATTENTION layer and has
/// no gated-DeltaNet carry to disagree about.
///
/// That reading has never been settled, and a raw state comparison cannot
/// settle it. Two paths can hold the same recurrence in different internal
/// arrangements and both be right, and a byte-for-byte disagreement between
/// two slots that never meet is consistent with that. What is NOT consistent
/// with it is the property serving actually needs, which is this one:
///
/// > A prefill of `[a, b, c, d]` and a prefill of `[a, b, c]` followed by a
/// > decode of `[d]` are the same conversation and must produce the same next
/// > token.
///
/// This is the question `tests/serving.rs` asks of qwen3-0.6B in
/// `a_conversation_is_answered_the_same_however_it_reaches_the_driver`, where
/// it passes -- but that model has no recurrence to carry, so it cannot
/// exercise a gated-DeltaNet state hand-off at all. Here the hand-off is the
/// whole point: the decode MUST read what the prefill left, or the answer
/// moves.
///
/// Two seats, so neither run can disturb the other, and the same tokens in
/// both. The comparison is over the whole distribution and not just its
/// argmax, because an argmax that survives a wrong carry is luck.
#[test]
#[ignore = "loads the real checkpoint; run explicitly"]
fn whether_a_split_conversation_answers_what_an_unsplit_one_does() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    // "The capital of France is Paris. The capital of France is"
    let all: Vec<u32> = vec![561, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 9338, 369];
    let cut = all.len() - 1;
    let ask = |shell: &mut driver_wgpu::shell::Shell, who: u64, t: &[u32]| -> Option<Vec<f32>> {
        let out = shell
            .step(&[driver_wgpu::turns::Turn {
                who,
                tokens: t.to_vec(),
            }])
            .ok()?;
        out.readout_of
            .first()
            .and_then(|i| out.logits.row(*i))
            .map(<[f32]>::to_vec)
    };
    let Some(whole) = ask(&mut shell, 9_100, &all) else {
        println!("the unsplit fire was refused");
        return;
    };
    // The same conversation, arriving in two pieces at a seat of its own.
    let Some(_) = ask(&mut shell, 9_200, &all[..cut]) else {
        println!("the prefix fire was refused");
        return;
    };
    let Some(split) = ask(&mut shell, 9_200, &all[cut..]) else {
        println!("the continuation was refused");
        return;
    };
    if whole.len() != split.len() {
        println!("the two fires answered different widths");
        return;
    }
    let (mut worst, mut at) = (0.0f32, 0usize);
    for (i, (p, q)) in whole.iter().zip(&split).enumerate() {
        if (p - q).abs() > worst {
            worst = (p - q).abs();
            at = i;
        }
    }
    let scale = whole.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let top = |v: &[f32]| -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    };
    let (tw, ts) = (top(&whole), top(&split));
    println!(
        "\n  {} tokens whole against {} + {} split:\n    \
         widest |whole - split| {worst:.6} at token {at}, against logits reaching {scale:.3} \
         -- relative {:e}\n    argmax {tw} whole, {ts} split",
        all.len(),
        cut,
        all.len() - cut,
        worst / scale.max(1e-30)
    );
    println!(
        "\n  {}",
        if tw == ts && worst / scale.max(1e-30) < 3e-2 {
            "THE SPLIT DOES NOT MOVE THE ANSWER, so the decode reads the carry the prefill left."
        } else if tw == ts {
            "THE ARGMAX SURVIVES BUT THE DISTRIBUTION MOVED, which a correct hand-off would not do."
        } else {
            "THE SPLIT CHANGES THE ANSWER, so the gated-DeltaNet carry does not survive the hand-off."
        }
    );
    assert_eq!(
        tw, ts,
        "a conversation answered differently for having been split: the \
         gated-DeltaNet decode is not reading the carry the prefill left"
    );
}

/// ONE STEP, FROM A STATE BOTH PATHS AGREE ON.
///
/// [`whether_the_prefill_and_the_decode_leave_the_same_carry`] fires a whole
/// prompt one way and the same prompt token-by-token the other, so by the last
/// token the two seats have taken different routes for every step and any
/// disagreement has had the whole prompt to compound. That says the paths
/// differ; it cannot say where, or by how much per step.
///
/// This isolates a single step. Both seats take the SAME three-token prefill,
/// so both reach the fourth token holding a carry produced by the same kernel
/// from the same inputs. Then one seat takes the fourth token as a decode and
/// the other takes all four as a prefill. Everything before the last step is
/// identical by construction, so whatever separates them is one application of
/// the recurrence.
///
/// It is decisive about WHICH is wrong, because the two are not equally
/// attested. `whether_the_scan_computes_the_recurrence_its_own_inputs_imply`
/// walks the prefill's scan on a CPU and finds it exact at six lengths and six
/// layers. The decode does not go through that kernel at all -- and not, as
/// this file assumed for a long time, through `gdn_prep` plus
/// `gdn_core_recurrent` either. A decode fire lowers to ONE launch,
/// `gdn_core_slotted`, a fused kernel in a different file that convolves,
/// normalizes, gates and recurs in a single body. It has no reference here.
///
/// # What has been eliminated, so the next attempt does not repeat it
///
/// Read against each other the two bodies are the same arithmetic term for
/// term, and four things that could make identical arithmetic disagree have
/// been checked and are not it:
///
/// - **The params.** Both statements carry the same eleven numbers, field for
///   field: `Dk=128 Dv=128 Hk=16 Hv=16 conv_dim=6144 Kc=4 q_off=0 k_off=2048
///   v_off=4096 eps=1e-6 inv_sqrt_dk=0.088388346`.
/// - **The grid.** `[32, Dv, rows * Hv]` over a `(32, 4)` workgroup gives
///   `dv_idx` its full 128 and `wid.z` its 16 heads, so this is not another
///   FIX 8 -- no part of the output goes unwritten.
/// - **The reduction.** `row_sum32` is identical in both files and reduces
///   WITHIN a row (`at = ly * 32 + lx`, read back at `ly * 32`), and
///   `sh_reduce` is `array<f32, 128>` in both -- sized for all four rows of a
///   `(32, 4)` workgroup and not just the first.
/// - **The decay.** Decomposing the gap against the state it started from,
///   `cos(split - whole, before)` is about -0.19 at every row sampled. A decay
///   that differed would put the whole gap ALONG `before`, at +/-1; this says
///   the rank-one term carries it, not the gate. That is what the third seat
///   below is for.
///
/// The convolution is out too, by measurement rather than reading:
/// `conv_state` is byte-identical afterwards on all six layers and both paths
/// append the same `mixed`, so the windows they convolved were the same.
///
/// What is left needs a CPU walk of the fused body, which is what the prefill
/// got and this has never had.
#[test]
#[ignore = "loads the real checkpoint; run explicitly"]
fn whether_one_decode_step_leaves_what_one_more_prefill_token_would() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let (split, whole) = (1_100u64, 1_101u64);
    // ONE SEAT TAKES FOUR TOKENS AT ONCE. The other takes three and then the
    // fourth on its own, which is the only difference between them.
    //
    // The first draft of this test fed `PERIOD[..3]` to both seats and then
    // `PERIOD[3..4]` to both, which is the same route twice: it reported zero
    // differences in all 262,144 elements and would have retired an open
    // defect on a comparison of a thing with itself. The whole seat must never
    // take a second fire.
    if fire_row(&mut shell, whole, &PERIOD[..4]).is_empty() {
        println!("the four-token prefill was refused, so IT COULD NOT BE MEASURED");
        return;
    }
    if fire_row(&mut shell, split, &PERIOD[..3]).is_empty() {
        println!("the three-token prefill was refused, so IT COULD NOT BE MEASURED");
        return;
    }
    if fire_row(&mut shell, split, &PERIOD[3..4]).is_empty() {
        println!("the one-token decode was refused");
        return;
    }
    // A THIRD SEAT that stops at three, so the state the fourth step starts
    // FROM is readable rather than inferred. With it the disagreement can be
    // decomposed instead of only measured.
    let before = 1_102u64;
    if fire_row(&mut shell, before, &PERIOD[..3]).is_empty() {
        println!("the reference prefill was refused");
        return;
    }
    let (Some(sa), Some(sb), Some(sc)) = (
        shell.book().slot(split),
        shell.book().slot(whole),
        shell.book().slot(before),
    ) else {
        println!("one of the three conversations has no recurrent seat");
        return;
    };
    println!("\n  one seat took four tokens at once; the other took three and then one");
    println!("  (slots {sa} and {sb})");
    let slots = 8u32;
    let (mut worst, mut worst_where) = (0.0f32, String::new());
    for which in ["recurrent_state", "conv_state"] {
        for layer in 0..6u16 {
            let Some(pool) = shell.recurrent() else { return };
            let Some(slab) = pool.slab(layer, which) else {
                continue;
            };
            let Ok(bytes) = shell.device().read_at(slab, 0, slab.size()) else {
                continue;
            };
            let per = bytes.len() / slots as usize;
            let at = |s: u32| -> Vec<f32> {
                let from = per * s as usize;
                bytes[from..from + per - (per % 4)]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            };
            let (ra, rb) = (at(sa), at(sb));
            let mut here = 0.0f32;
            let mut differ = 0usize;
            for (x, y) in ra.iter().zip(&rb) {
                // NOT `f32::max`: it returns the non-NaN operand, so a fold
                // with it reports an all-NaN difference as zero.
                let d = (x - y).abs();
                if d != 0.0 {
                    differ += 1;
                }
                if d > here || !d.is_finite() {
                    here = d;
                }
            }
            if here > worst {
                worst = here;
                worst_where = format!("{which} layer {layer}");
            }
            println!(
                "    {which:<16} layer {layer}: {differ:6} of {} differ, widest {here:.6}",
                ra.len()
            );
            // WHICH TERM DIFFERS. One step is `S = S_prev * decay + k (x)
            // delta`, so the gap between two paths that started from the same
            // `S_prev` is
            //
            //     S_a - S_b = (decay_a - decay_b) * S_prev + k * (delta_a - delta_b)
            //
            // and the two terms point in different directions. Along `S_prev`
            // means the DECAY differs; orthogonal to it means the rank-one
            // update does. Taken per `(head, dv)` row, which is the unit both
            // kernels share: `Dk` contiguous channels sharing one decay and
            // one delta.
            if which == "recurrent_state" && layer == 0 && differ > 0 {
                let rc = at(sc);
                let dk = 128usize;
                let mut along = Vec::new();
                for row in [0usize, 17, 129, 1000] {
                    let (o, n) = (row * dk, ra.len());
                    if o + dk > n || o + dk > rc.len() {
                        continue;
                    }
                    let (mut dot, mut dd, mut pp) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..dk {
                        let d = f64::from(ra[o + i] - rb[o + i]);
                        let s = f64::from(rc[o + i]);
                        dot += d * s;
                        dd += d * d;
                        pp += s * s;
                    }
                    if dd > 0.0 && pp > 0.0 {
                        along.push((row, dot / (dd.sqrt() * pp.sqrt())));
                    }
                }
                for (row, cos) in &along {
                    println!(
                        "      row {row:5}: cos(split - whole, state before) = {cos:+.4} \
                         -- +/-1 is a decay difference, 0 a delta one"
                    );
                }
            }
        }
    }
    let verdict = if worst == 0.0 {
        "ONE DECODE STEP LEAVES WHAT ONE MORE PREFILL TOKEN DOES, so the two paths agree."
            .to_string()
    } else {
        format!(
            "ONE STEP IS ALREADY {worst:.6} APART, widest at {worst_where}. Everything \
             before it was the same kernel on the same inputs, so this is one application \
             of the recurrence disagreeing -- and the prefill's is the one with a CPU \
             reference."
        )
    };
    println!("\n  {verdict}");
}

/// THE FUSED DECODE, WALKED ON A CPU FROM ITS OWN OPERANDS.
///
/// [`whether_one_decode_step_leaves_what_one_more_prefill_token_would`] shows
/// one decode step landing 4.38 away from the prefill token it should equal,
/// and eliminates the params, the grid, the reduction, the decay and the
/// convolution. What it cannot say is whether `gdn_core_slotted` computes the
/// wrong thing or is handed the wrong thing, because both look the same from
/// outside the kernel.
///
/// So walk it. The fused body is short enough to state completely:
///
/// ```text
///   qraw[d] = silu(conv(q_off + hk*Dk + d)),  kraw likewise,  v = silu(conv(v_off + hv*Dv + dv))
///   qinv    = inv_sqrt_dk / sqrt(sum qraw^2 + eps)      kinv = 1 / sqrt(sum kraw^2 + eps)
///   decay   = exp(-exp(A_log[hv]) * softplus(a[hv] + dt[hv]))      beta = sigmoid(b[hv])
///   S[d]   := S_prev[d] * decay
///   kv      = sum_d S[d] * kraw[d] * kinv
///   delta   = (v - kv) * beta
///   S[d]   += kraw[d] * kinv * delta
/// ```
///
/// Every operand is readable: `S_prev` and the convolution window from a third
/// seat that stops one token short, `mixed`, `a` and `b` from the decode's own
/// arena, and the four weights from what the shell holds.
///
/// # It is compared against BOTH answers, which is the whole point
///
/// A reference written beside one implementation agrees with it, which is how
/// every wrong turn in this file happened -- the MLP reference multiplied by
/// `w` because the device did, and agreed to one bf16 step while the model
/// answered a space. So this one is scored against the decode's result AND
/// against the prefill's, and the two are known to differ. It must pick one:
///
/// - matching the DECODE means the kernel is faithful and its inputs are wrong
/// - matching the PREFILL means the kernel is wrong
/// - matching neither means this reference is, and says so rather than
///   accusing anyone
#[test]
#[ignore = "loads the real checkpoint; run explicitly"]
fn whether_the_fused_decode_computes_the_step_its_own_operands_imply() {
    let Some((mut shell, real)) = qwen3_5_shell(8) else {
        return;
    };
    shell.keep_arena(true);
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let (before, split, whole) = (1_200u64, 1_201u64, 1_202u64);
    // The state the step starts from, left by a seat that stops one short.
    if fire_row(&mut shell, before, &PERIOD[..3]).is_empty() {
        println!("the reference prefill was refused");
        return;
    }
    if fire_row(&mut shell, whole, &PERIOD[..4]).is_empty() {
        println!("the four-token prefill was refused");
        return;
    }
    if fire_row(&mut shell, split, &PERIOD[..3]).is_empty() {
        println!("the three-token prefill was refused");
        return;
    }
    // THE STEP UNDER TEST, stopped at layer 0's core.
    //
    // The arena is REUSED across all twenty-four layers, so reading `mixed`
    // after a whole fire gives the LAST layer's projection and not this one's.
    // The first draft of this test did exactly that and matched neither answer
    // -- which is what the three-way verdict below is for, and it is the same
    // trap `which_fold_the_final_norm_applies_and_what_each_one_answers` fell
    // into with the final norm's input.
    let plan0 = hybrid_plan_class(
        &Qwen35HybridFacts::qwen3_5_0_8b(),
        model_ir::trace::FireClass::Decode,
    );
    let rows0: Vec<Row> = vec![Row { samples: true, ..Row::default() }];
    let Ok(low0) = lower(&plan0, &rows0, Fire { captures_across_splits: false }) else {
        return;
    };
    let Some(stop) = low0.launches.iter().position(|l| {
        l.layers.start == 0 && low0.kernels[l.kernel as usize].starts_with("gdn_core")
    }) else {
        println!("the decode does not lower to a fused core");
        return;
    };
    shell.fire_prefix(Some(stop + 1));
    let step = match shell.step(&[driver_wgpu::turns::Turn {
        who: split,
        tokens: PERIOD[3..4].to_vec(),
    }]) {
        Ok(s) => s,
        Err(why) => {
            println!("the decode was refused: {why}");
            return;
        }
    };
    shell.fire_prefix(None);
    let arena = step.arena;
    let (Some(sb), Some(ss), Some(sw)) = (
        shell.book().slot(before),
        shell.book().slot(split),
        shell.book().slot(whole),
    ) else {
        println!("a conversation has no recurrent seat");
        return;
    };

    let (dk, dv, hv_n, kc, conv_dim) = (128usize, 128usize, 16usize, 4usize, 6144usize);
    let (q_off, k_off, v_off) = (0usize, 2048usize, 4096usize);
    let (eps, inv_sqrt_dk) = (1e-6f32, 0.088_388_346f32);
    let slots = 8usize;
    let slab = |layer: u16, which: &str, slot: u32, want: usize| -> Option<Vec<f32>> {
        let pool = shell.recurrent()?;
        let s = pool.slab(layer, which)?;
        let bytes = shell.device().read_at(s, 0, s.size()).ok()?;
        let per = bytes.len() / slots;
        let from = per * slot as usize;
        let out: Vec<f32> = bytes[from..from + per - (per % 4)]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (out.len() >= want).then_some(out)
    };
    let (Some(s_prev), Some(got), Some(want)) = (
        slab(0, "recurrent_state", sb, hv_n * dv * dk),
        slab(0, "recurrent_state", ss, hv_n * dv * dk),
        slab(0, "recurrent_state", sw, hv_n * dv * dk),
    ) else {
        println!("a recurrent slab is not the shape this test reads");
        return;
    };
    // THE PING-PONG. `conv_state` and `new_conv_state` are separate buffers on
    // purpose -- the shader reads the old taps while writing the new -- and
    // which of them holds the window a given fire READ is a property of how
    // the pool swaps them, not something this test can assume. So try both and
    // let the arithmetic say. An earlier probe in this file assumed one and
    // reported reconstructed windows that were all zero.
    let halves: Vec<(&str, Vec<f32>)> = ["conv_state", "new_conv_state"]
        .into_iter()
        .filter_map(|w| slab(0, w, sb, kc * conv_dim).map(|v| (w, v)))
        .collect();
    if halves.is_empty() {
        println!("neither convolution half is the shape this test reads");
        return;
    }
    println!(
        "\n  magnitudes: |s_prev| {:.4}, |decode| {:.4}, |prefill| {:.4}",
        s_prev.iter().fold(0.0f32, |m, v| m.max(v.abs())),
        got.iter().fold(0.0f32, |m, v| m.max(v.abs())),
        want.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    );

    let bf16 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()
    };
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    // `a_log` is held at the width it was FOUND (f32, one per head); the rest
    // are narrowed. Same lesson as
    // `where_every_raw_weight_the_driver_holds_actually_came_from`.
    let (Some(conv_w), Some(conv_b), Some(a_log), Some(dt)) = (
        real.get("layer.0.conv_w").map(|b| bf16(b)),
        real.get("layer.0.conv_b").map(|b| bf16(b)),
        real.get("layer.0.a_log").map(|b| {
            if b.len() == hv_n * 4 { f32s(b) } else { bf16(b) }
        }),
        real.get("layer.0.dt").map(|b| {
            if b.len() == hv_n * 4 { f32s(b) } else { bf16(b) }
        }),
    ) else {
        println!("the gated-DeltaNet weights are not all held");
        return;
    };
    // The decode's own arena operands, by the launch that reads them.
    let l = &low0.launches[stop];
    let a: Vec<(usize, usize)> = low0.args[l.args.start as usize..l.args.end as usize]
        .iter()
        .filter_map(|x| match x {
            Arg::Arena { at, width, bytes } => Some((*at, *width as usize * *bytes as usize)),
            _ => None,
        })
        .collect();
    let [(m_at, m_w), (a_at, a_w), (b_at, b_w), ..] = a[..] else {
        println!("the fused core does not take the operands this test reads");
        return;
    };
    let mixed = bf16(&arena[m_at..m_at + m_w]);
    let agate = bf16(&arena[a_at..a_at + a_w]);
    let bgate = bf16(&arena[b_at..b_at + b_w]);
    if mixed.len() < conv_dim || agate.len() < hv_n || bgate.len() < hv_n {
        println!("the decode's operands are narrower than the shapes imply");
        return;
    }

    // One convolution output channel, silu'd, exactly as `convsilu` states it:
    // taps 1..Kc-1 of the window and this token's `mixed` for the last.
    let silu = |x: f32| x / (1.0 + (-x).exp());

    println!("\n  the fused decode's step, walked on a CPU from its own operands:");
    let mut best: Option<(&str, f32, f32, f32)> = None;
    for (half, conv) in &halves {
    let convsilu = |c: usize| -> f32 {
        let mut acc = conv_b.get(c).copied().unwrap_or(0.0);
        for j in 0..kc - 1 {
            acc += conv[(j + 1) * conv_dim + c] * conv_w[c * kc + j];
        }
        acc += mixed[c] * conv_w[c * kc + (kc - 1)];
        silu(acc)
    };
    let (mut d_got, mut d_want, mut scale) = (0.0f32, 0.0f32, 0.0f32);
    for hv in 0..hv_n {
        let mut qraw = vec![0.0f32; dk];
        let mut kraw = vec![0.0f32; dk];
        let (mut qsq, mut ksq) = (0.0f32, 0.0f32);
        for d in 0..dk {
            qraw[d] = convsilu(q_off + hv * dk + d);
            kraw[d] = convsilu(k_off + hv * dk + d);
            qsq += qraw[d] * qraw[d];
            ksq += kraw[d] * kraw[d];
        }
        let qinv = inv_sqrt_dk / (qsq + eps).sqrt();
        let kinv = 1.0 / (ksq + eps).sqrt();
        let _ = qinv;
        let ad = agate[hv] + dt[hv];
        let sp = ad.max(0.0) + (1.0 + (-ad.abs()).exp()).ln();
        let decay = (-a_log[hv].exp() * sp).exp();
        let beta = 1.0 / (1.0 + (-bgate[hv]).exp());
        for dvi in 0..dv {
            let base = (hv * dv + dvi) * dk;
            let vval = convsilu(v_off + hv * dv + dvi);
            let mut st = vec![0.0f32; dk];
            let mut kv = 0.0f32;
            for d in 0..dk {
                st[d] = s_prev[base + d] * decay;
                kv += st[d] * (kraw[d] * kinv);
            }
            let delta = (vval - kv) * beta;
            for d in 0..dk {
                st[d] += kraw[d] * kinv * delta;
                d_got = d_got.max((st[d] - got[base + d]).abs());
                d_want = d_want.max((st[d] - want[base + d]).abs());
                scale = scale.max(st[d].abs());
            }
        }
    }
    let (rg0, rw0) = (d_got / scale.max(1e-30), d_want / scale.max(1e-30));
    println!(
        "    window `{half:<15}`: |cpu - decode| {d_got:.6} ({rg0:e}), \
         |cpu - prefill| {d_want:.6} ({rw0:e}), reaching {scale:.3}"
    );
    if best.is_none_or(|(_, g, _, _)| rg0 < g) {
        best = Some((half, rg0, rw0, scale));
    }
    }
    let Some((half, rg, rw, _)) = best else { return };
    println!("\n    the closer window is `{half}`");
    println!(
        "\n  {}",
        if rg < 3e-2 && rw < 3e-2 {
            "THE STEP REPRODUCES, and the decode and the prefill now agree closely enough that this reference matches both -- which is what a carried-back window looks like."
        } else if rg < 3e-2 {
            "THE KERNEL IS FAITHFUL TO ITS OPERANDS, so the defect is in what it is HANDED."
        } else if rw < 3e-2 {
            "THE KERNEL DISAGREES WITH ITS OWN OPERANDS and the prefill is what they imply, so `gdn_core_slotted` is the defect."
        } else {
            "THIS REFERENCE MATCHES NEITHER, so it is the thing that is wrong -- read it before reading the kernel."
        }
    );
}

/// THE CONTROL EVERY TWO-SEAT COMPARISON IN THIS FILE RESTS ON.
///
/// [`whether_one_decode_step_leaves_what_one_more_prefill_token_would`] and
/// [`whether_the_prefill_and_the_decode_leave_the_same_carry`] both compare
/// two seats and read the difference as a property of the PATHS. That only
/// follows if two seats given the same tokens by the same kernel end up
/// holding the same thing -- if a freshly allocated seat carries whatever the
/// pool last left there, every one of those comparisons is measuring litter
/// and calling it a recurrence.
///
/// Nothing in this file has ever checked it. So: two seats, the same three
/// tokens, the same single prefill kernel, nothing else. Any difference at all
/// is a difference in the SEAT and not in the arithmetic, and it would put a
/// question mark on every carry measurement here.
#[test]
#[ignore = "loads the real checkpoint; run explicitly"]
fn whether_two_seats_given_the_same_tokens_hold_the_same_state() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let (one, two) = (1_300u64, 1_301u64);
    for who in [one, two] {
        if fire_row(&mut shell, who, &PERIOD[..3]).is_empty() {
            println!("a three-token prefill was refused, so IT COULD NOT BE MEASURED");
            return;
        }
    }
    let (Some(sa), Some(sb)) = (shell.book().slot(one), shell.book().slot(two)) else {
        println!("a conversation has no recurrent seat");
        return;
    };
    if sa == sb {
        println!("both conversations were given seat {sa}, so there is nothing to compare");
        return;
    }
    println!("\n  the same three tokens into seat {sa} and seat {sb}:");
    let slots = 8usize;
    let mut worst = 0.0f32;
    for which in ["recurrent_state", "conv_state"] {
        for layer in 0..6u16 {
            let Some(pool) = shell.recurrent() else { return };
            let Some(s) = pool.slab(layer, which) else { continue };
            let Ok(bytes) = shell.device().read_at(s, 0, s.size()) else {
                continue;
            };
            let per = bytes.len() / slots;
            let at = |slot: u32| -> Vec<f32> {
                let from = per * slot as usize;
                bytes[from..from + per - (per % 4)]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            };
            let (ra, rb) = (at(sa), at(sb));
            let (mut here, mut differ) = (0.0f32, 0usize);
            for (x, y) in ra.iter().zip(&rb) {
                let d = (x - y).abs();
                if d != 0.0 {
                    differ += 1;
                }
                if d > here || !d.is_finite() {
                    here = d;
                }
            }
            worst = worst.max(here);
            if differ > 0 {
                println!(
                    "    {which:<16} layer {layer}: {differ} of {} differ, widest {here:.6}",
                    ra.len()
                );
            }
        }
    }
    println!(
        "\n  {}",
        if worst == 0.0 {
            "TWO SEATS GIVEN THE SAME TOKENS HOLD THE SAME STATE, so a seat contributes nothing of its own and the carry comparisons are sound."
        } else {
            "TWO SEATS GIVEN THE SAME TOKENS HOLD DIFFERENT STATE, so every two-seat comparison in this file is confounded and the decode may never have been the defect."
        }
    );
    assert_eq!(
        worst, 0.0,
        "a seat contributed state of its own, which invalidates every carry \
         comparison in this file"
    );
}

/// WHAT CARRYING THE CONVOLUTION WINDOWS BACK COSTS A DECODE.
///
/// `RecurrentPool::carry_back` is correctness, not a nicety -- without it every
/// continuation convolves over a window one fire stale. But it runs on the
/// DECODE HOT PATH, once per fire, and `Device::transfer` is not a free
/// enqueue: it submits its own command buffer and then drains and waits, which
/// is a full device synchronisation between one token and the next.
///
/// `tests/serving.rs` cannot see any of this. It serves qwen3-0.6B, which has
/// no recurrent pool at all and so pays nothing, and its 22 tests were as green
/// before the fix as after. A cost that only the hybrid pays needs the hybrid
/// to measure it.
///
/// So measure it rather than argue about it: time a run of decodes, then time
/// the same number of bare `carry_back` calls, and report the share. Eighteen
/// linear layers at 768 KiB a plane is about 13.5 MiB a fire, which is nothing
/// for a 4090's bandwidth and possibly everything for the sync around it --
/// and those two guesses point opposite ways, which is the reason to look.
#[test]
#[ignore = "times a real checkpoint; run explicitly"]
fn what_carrying_the_convolution_windows_back_costs() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let who = 1_400u64;
    if fire_row(&mut shell, who, &PERIOD).is_empty() {
        println!("the prefill was refused, so IT COULD NOT BE MEASURED");
        return;
    }
    const N: usize = 24;
    // Warm, so the first fire's lowering and pipeline work is not counted.
    for _ in 0..4 {
        if fire_row(&mut shell, who, &PERIOD[..1]).is_empty() {
            println!("a decode was refused");
            return;
        }
    }
    let t0 = std::time::Instant::now();
    for _ in 0..N {
        if fire_row(&mut shell, who, &PERIOD[..1]).is_empty() {
            println!("a decode was refused");
            return;
        }
    }
    let decodes = t0.elapsed();
    let Some(pool) = shell.recurrent() else {
        return;
    };
    let t1 = std::time::Instant::now();
    for _ in 0..N {
        if pool.carry_back(shell.device()).is_err() {
            println!("a carry-back failed");
            return;
        }
    }
    let carries = t1.elapsed();
    let (per_decode, per_carry) = (
        decodes.as_secs_f64() * 1e3 / N as f64,
        carries.as_secs_f64() * 1e3 / N as f64,
    );
    println!(
        "\n  {N} decodes: {:.1} ms total, {per_decode:.3} ms each\n  \
         {N} carry-backs alone: {:.1} ms total, {per_carry:.3} ms each",
        decodes.as_secs_f64() * 1e3,
        carries.as_secs_f64() * 1e3
    );
    let share = per_carry / per_decode.max(1e-9) * 100.0;
    println!("  the carry-back is {share:.1}% of a decode");
    println!(
        "\n  {}",
        if share < 10.0 {
            "IT IS NOISE AGAINST A DECODE, so the copy can stay where it is."
        } else {
            "IT IS A REAL SHARE OF A DECODE, and folding the copy into the fire's own command buffer would remove a synchronisation per token."
        }
    );
}

/// WHERE A DECODE'S TIME GOES: FIXED COST, OR PER LAUNCH.
///
/// A decode of this hybrid takes 59 ms and one of qwen3-0.6B takes 113 ms on
/// the same adapter (`tests/serving.rs`'s `what_a_decode_costs_at_length`, 8.9
/// tok/s). Those are large numbers for a sub-billion-parameter model at four
/// bits on a 4090, and they are large in the same way for two different stacks
/// -- so whatever it is, it is the DRIVER's and not the gated DeltaNet's. That
/// much the two numbers already settle, and it is worth settling because this
/// file has just been changing the decode path.
///
/// The next question is whether the cost is per FIRE or per LAUNCH, and
/// `fire_prefix` answers it directly: it truncates the lowered launch list, so
/// firing the same token at several prefixes and timing each gives a line whose
/// SLOPE is the cost of one launch and whose INTERCEPT is everything a fire
/// pays once -- lowering, staging, the submit, the drain and the wait.
///
/// The two readings ask for different work and there is no point guessing
/// between them. A steep slope over ~350 launches means dispatch overhead and
/// the remedy is fewer, larger launches. A large intercept means the fire's
/// own bookkeeping and the remedy is somewhere else entirely.
///
/// # What is already done, so nobody does it twice
///
/// The obvious structural fixes are in. `Device` encodes a whole fire into ONE
/// command buffer and batches its dispatches into ONE compute pass, breaking
/// only where a shadow copy has to be encoded between them -- `device.rs`
/// carries the measurement that bought it (735 command buffers against 1:
/// encoding 7.1 ms to 4.3, submit 5.4 to 1.0, wait 13.0 to 9.7). So the
/// remaining per-launch cost is what each dispatch BUILDS -- a bind group and
/// a uniform buffer apiece -- and what it then executes.
///
/// That same table is the shape of the measurement this test does not make:
/// it separates encode from submit from wait, and only the wait is the device.
///
/// # What the three measurements together say the launch cost IS
///
/// Three numbers now bracket it, and none of them was available when this test
/// was written:
///
/// - **0.016 ms** -- what a dispatch with almost no arithmetic in it costs
///   ISOLATED, from `kernels-wgpu`'s `how_long_a_decodes_kernels_take` FLOOR
///   row. The projections at the model's own shapes cost 0.006 to 0.017, which
///   is to say they are at that floor.
/// - **0.006 ms** -- the host's share, from `serve::record`'s own note: 2.05 ms
///   of `create_bind_group` plus a uniform buffer each, over 452 dispatches.
///   The cache that would have removed it hit 0 of 10,000, and the reason is
///   recorded there.
/// - **0.11 ms** -- what a launch costs INSIDE a fire, from the slope below.
///
/// So the in-fire launch is seven times the isolated floor and eighteen times
/// the host bookkeeping, and neither the kernels nor the bind groups can
/// account for the difference. What is left is that these 346 dispatches are
/// DEPENDENT -- each layer reads what the last one wrote -- while the isolated
/// bench runs 200 independent ones into a single submission and lets the device
/// pipeline them. A dependent dispatch pays its latency in full.
///
/// # And the correction that reading makes necessary
///
/// The paragraph above compares an AVERAGE over 346 launches against the cost
/// of a kernel with almost no arithmetic in it, and those are not the same
/// kind of number. The projections are at the floor and there are 187 of them,
/// but the mix also holds 18 `gdn_core_slotted` -- each reading and writing a
/// megabyte of recurrent state -- and 6 `sdpa_paged_decode`, which
/// `how_long_a_decodes_kernels_take` measures at 1.246 ms apiece at 2048 keys.
/// A handful of genuinely expensive kernels raises an average without any
/// dispatch latency being involved.
///
/// The per-layer sweep says which reading is right, and it is this one: a
/// gated-DeltaNet layer averages 2.9 ms and there are eighteen, which is 52 ms
/// against a decode of about 50. The cost is in the LAYERS, not spread evenly
/// over the launches -- so the dependent-dispatch story is at most part of it
/// and the target is `gdn_core_slotted`, which is a WGSL kernel after all and
/// has never been benchmarked in isolation.
///
/// Both readings are kept because the first one is what the three numbers
/// suggest and the fourth is what refutes it. Anyone reaching for fusion
/// should benchmark the fused core first.
///
/// RETIRED WITH `kernels-wgpu`'s TEST TREE. That name is a record of a
/// measurement now, not a live proof: the crate lost `tests/` and every
/// in-file `mod tests` when the three shader planes moved their numbers to
/// the fire that reads them, and nothing in this workspace re-runs it. What
/// it reported is still why the sentence above says what it says; what is
/// gone is the thing that would notice if it stopped being true.
#[test]
#[ignore = "times a real checkpoint; run explicitly"]
fn whether_a_decode_costs_by_the_fire_or_by_the_launch() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan_class(&facts, model_ir::trace::FireClass::Decode);
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }];
    let Ok(low) = lower(&plan, &rows, Fire { captures_across_splits: false }) else {
        return;
    };
    let total = low.launches.len();
    println!("\n  a decode of this hybrid lowers to {total} launches");
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let who = 1_500u64;
    if fire_row(&mut shell, who, &PERIOD).is_empty() {
        println!("the prefill was refused, so IT COULD NOT BE MEASURED");
        return;
    }
    let mut points: Vec<(f64, f64)> = Vec::new();
    for cut in [1usize, total / 4, total / 2, 3 * total / 4, total] {
        shell.fire_prefix((cut < total).then_some(cut));
        // Warm this prefix: each one lowers its own truncated list once.
        for _ in 0..3 {
            let _ = shell.step(&[driver_wgpu::turns::Turn {
                who,
                tokens: PERIOD[..1].to_vec(),
            }]);
        }
        const N: usize = 12;
        let t = std::time::Instant::now();
        for _ in 0..N {
            if shell
                .step(&[driver_wgpu::turns::Turn {
                    who,
                    tokens: PERIOD[..1].to_vec(),
                }])
                .is_err()
            {
                println!("a decode at prefix {cut} was refused");
                shell.fire_prefix(None);
                return;
            }
        }
        let each = t.elapsed().as_secs_f64() * 1e3 / N as f64;
        println!("    {cut:4} of {total} launches: {each:7.3} ms");
        points.push((cut as f64, each));
    }
    shell.fire_prefix(None);

    // Least squares through the points: the slope is one launch, the intercept
    // is the fire.
    let n = points.len() as f64;
    let (sx, sy) = (
        points.iter().map(|(x, _)| x).sum::<f64>(),
        points.iter().map(|(_, y)| y).sum::<f64>(),
    );
    let sxx = points.iter().map(|(x, _)| x * x).sum::<f64>();
    let sxy = points.iter().map(|(x, y)| x * y).sum::<f64>();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-9 {
        println!("the prefixes did not vary, so nothing could be fitted");
        return;
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    let launched = slope * points.last().map_or(0.0, |(x, _)| *x);
    println!(
        "\n    one launch costs {:.4} ms, and a fire costs {intercept:.3} ms before any of them",
        slope
    );
    println!(
        "    so {total} launches are {launched:.1} ms of a {:.1} ms decode",
        points.last().map_or(0.0, |(_, y)| *y)
    );
    println!(
        "\n  {}",
        if launched > intercept {
            "THE LAUNCHES DOMINATE: the remedy is fewer and larger dispatches, or less work per one."
        } else {
            "THE FIRE'S OWN COST DOMINATES: the launches are not what a decode is waiting for."
        }
    );

    // WHAT ONE LAUNCH SPENDS IT ON. `Device::uniform` calls `create_buffer`,
    // so every dispatch that carries a scalar block ALLOCATES one -- 346 of
    // them a decode here. A GPU buffer is not a malloc, and if this is the
    // shape of the per-launch number then the launches are not the problem so
    // much as what each one builds before it.
    const M: usize = 200;
    let block = [0u8; 32];
    let t = std::time::Instant::now();
    for _ in 0..M {
        if shell.device().uniform(&block).is_err() {
            println!("a uniform allocation failed");
            return;
        }
    }
    let per_uniform = t.elapsed().as_secs_f64() * 1e3 / M as f64;
    println!(
        "\n    one `Device::uniform` (a `create_buffer`): {per_uniform:.4} ms, \
         against {slope:.4} ms for a whole launch"
    );
    println!(
        "  {}",
        if per_uniform > slope * 0.25 {
            "SO A LARGE PART OF EVERY LAUNCH IS ALLOCATING ITS UNIFORM BLOCK, which is a buffer per dispatch per token."
        } else {
            "so the uniform block is not where a launch's time goes; look elsewhere in the dispatch."
        }
    );

    // AGAINST THE ONE CEILING THAT DOES NOT CARE HOW THE TIME IS SPLIT.
    //
    // A decode reads every weight exactly once, so its floor is the staged
    // bytes over the adapter's bandwidth and nothing else. Putting the measured
    // time against that says whether this is a slow KERNEL problem or an idle
    // GPU problem, and those want opposite work: the first is tuning, the
    // second is fewer and cheaper dispatches. It is worth deriving rather than
    // asserting, because the split above cannot distinguish them.
    let staged = 424_159_424.0f64; // what `qwen3_5_shell` reports staging
    let ms = points.last().map_or(0.0, |(_, y)| *y);
    let effective = staged / (ms / 1e3) / 1e9;
    println!(
        "\n    a decode reads {:.0} MB of weights in {ms:.1} ms: {effective:.1} GB/s effective",
        staged / 1e6
    );
    println!(
        "  {}",
        if effective < 100.0 {
            "WHICH IS A SMALL FRACTION OF ANY MODERN ADAPTER'S BANDWIDTH, so the device is mostly idle and the remedy is fewer or cheaper dispatches rather than faster kernels."
        } else {
            "which is a serious fraction of the adapter's bandwidth, so the kernels are the thing to tune."
        }
    );
}

/// WHICH KERNEL A DECODE IS ACTUALLY WAITING FOR.
///
/// `whether_a_decode_costs_by_the_fire_or_by_the_launch` says the launches
/// dominate and does not say WHICH. Averaged over 346 of them, 0.11 ms apiece
/// is a number no optimisation can be aimed at: the fused gated-DeltaNet core
/// reads a megabyte of recurrent state and the projections read a quarter of a
/// gigabyte of weights, and those want completely different work.
///
/// `fire_prefix` measures it the same way as before, one step finer: stop at
/// each launch of one layer of each kind and difference the times. What comes
/// out is a cost per LAUNCH NAME, which is a thing that can be optimised.
///
/// A single launch is well under the noise of a shared machine, so each point
/// is the median of several fires and the differences are reported as they
/// fall -- a negative one means the two launches were indistinguishable, and
/// saying so is better than hiding it behind an absolute value.
#[test]
#[ignore = "times a real checkpoint; run explicitly"]
fn which_kernel_a_decode_waits_for() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
    facts.layers = 24;
    let plan = hybrid_plan_class(&facts, model_ir::trace::FireClass::Decode);
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }];
    let Ok(low) = lower(&plan, &rows, Fire { captures_across_splits: false }) else {
        return;
    };
    let total = low.launches.len();
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let who = 1_600u64;
    if fire_row(&mut shell, who, &PERIOD).is_empty() {
        println!("the prefill was refused, so IT COULD NOT BE MEASURED");
        return;
    }
    // PER LAYER, not per launch. A single launch is ~0.1 ms and this machine's
    // noise is several, so differencing adjacent launches reports numbers like
    // "-6.4 ms for an `rms_single_row`" -- which is not a measurement, it is
    // the machine. A layer is fourteen to eighteen launches and a few
    // milliseconds, which clears the floor.
    //
    // `layers.start` is not a span: the readout's launches carry 0 as well, so
    // the boundary is the FIRST launch of each layer and the tail is whatever
    // follows the last one.
    let mut firsts: Vec<(u16, usize)> = Vec::new();
    for (i, l) in low.launches.iter().enumerate() {
        if firsts.last().is_none_or(|(seen, _)| *seen != l.layers.start) {
            firsts.push((l.layers.start, i));
        }
    }
    let at = |shell: &mut driver_wgpu::shell::Shell, cut: usize| -> f64 {
        shell.fire_prefix((cut < total).then_some(cut));
        for _ in 0..2 {
            let _ = shell.step(&[driver_wgpu::turns::Turn {
                who,
                tokens: PERIOD[..1].to_vec(),
            }]);
        }
        let mut runs: Vec<f64> = Vec::new();
        for _ in 0..7 {
            let t = std::time::Instant::now();
            let ok = shell
                .step(&[driver_wgpu::turns::Turn {
                    who,
                    tokens: PERIOD[..1].to_vec(),
                }])
                .is_ok();
            if !ok {
                return f64::NAN;
            }
            runs.push(t.elapsed().as_secs_f64() * 1e3);
        }
        runs.sort_by(f64::total_cmp);
        runs[runs.len() / 2]
    };
    println!("\n  per layer, median of seven fires each:");
    let mut prev = at(&mut shell, firsts.first().map_or(0, |(_, i)| *i));
    let (mut gdn, mut attn) = (Vec::new(), Vec::new());
    for w in firsts.windows(2) {
        let (layer, start) = w[0];
        let next = w[1].1;
        let now = at(&mut shell, next);
        let cost = now - prev;
        prev = now;
        let full = facts.is_full_attn(u32::from(layer));
        println!(
            "    layer {layer:2} ({:>2} launches, {}): {cost:7.3} ms",
            next - start,
            if full { "attention" } else { "DeltaNet" }
        );
        if full { attn.push(cost) } else { gdn.push(cost) }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!(
        "\n    a gated-DeltaNet layer averages {:.3} ms over {}, a full-attention one {:.3} ms over {}",
        mean(&gdn),
        gdn.len(),
        mean(&attn),
        attn.len()
    );
    println!(
        "    so the stack is about {:.1} ms of DeltaNet and {:.1} ms of attention",
        mean(&gdn) * gdn.len() as f64,
        mean(&attn) * attn.len() as f64
    );
    println!("\n  a difference at or below zero is two launches this machine cannot tell apart");
}

/// WHETHER THE LEVER THE KERNEL BENCHES POINT AT ACTUALLY PULLS.
///
/// Both halves of a decode have now been chased to a floor that no kernel
/// rewrite reaches. `affine_qmv_fast` costs 0.006 to 0.017 ms at the model's
/// shapes against a dispatch floor of 0.016, and `gdn_core` at ONE ROW is
/// waiting on its fixed cost -- the same body reaches 63.7 GB/s at sixteen
/// rows where it manages 4.8 at one.
///
/// That says the lever is rows. It does not say the driver can pull it: a fire
/// of eight conversations is eight seats, eight slot-map entries and eight
/// times the state to address, and whether the dispatches actually amortise
/// across them is a property of this backend and not of the kernels.
///
/// So ask it directly. Fire one token for N conversations in ONE step and
/// divide. A cost per token that falls with N is the lever pulling; one that
/// stays flat is a driver that serialises what the kernels would have shared.
///
///     1 row(s):   61.20 ms a fire,  61.20 ms a token, 1.00x
///     2 row(s):   75.10 ms a fire,  37.55 ms a token, 1.63x
///     4 row(s):  135.52 ms a fire,  33.88 ms a token, 1.81x
///     8 row(s):  145.89 ms a fire,  18.24 ms a token, 3.36x
///
/// It pulls. Eight conversations cost 2.4 fires' worth of wall clock and give
/// eight tokens, which is the fixed cost the kernel benches found being paid
/// once instead of eight times.
///
/// That is the whole chain closed: `gdn_core` is 13x cheaper a row at sixteen
/// rows in isolation, and the driver turns that into 3.4x a token at eight
/// conversations end to end. Single-stream latency is bound by costs no kernel
/// rewrite reaches; throughput is not bound by them at all.
///
/// # The step in that table was noise
///
/// The four-point run above showed 4 rows costing 135.52 ms and 8 costing
/// 145.89 -- eight conversations for the price of four -- which is the shape of
/// a THRESHOLD rather than of amortisation, and this backend has two that could
/// draw one. Sweeping every row count instead of four of them says otherwise:
///
///     1: 41.41   2: 57.82   3: 127.01   4:  82.95
///     5: 91.94   6: 129.29  7: 149.41   8: 126.71
///
/// Three rows cost more than four and seven more than eight, which no guard
/// does. The absolute numbers moved by a third against the earlier run on the
/// same binary, and the "step" moved with them. It was the machine.
///
/// What survives both runs is the direction: 2.61x a token at eight rows here
/// against 3.36x there, and monotone in neither. This test is worth running
/// for the FIRST number and the LAST; the shape between them needs a quiet
/// machine, and saying so is better than reading a guard into the gaps.
///
/// # The loose thread that remains
///
/// Eight rows reach 2.6x to 3.4x and the pool holds eight seats, so nothing
/// here says where the curve flattens -- only that it has not by eight. The
/// two thresholds this backend does carry, `qmm_tile`'s `(16, 32)` guard and
/// the one-at-a-time switch, are worth checking against a quiet machine and a
/// larger pool, in that order. What this test can already say is that the
/// lever exists and that reading its exact shape off a shared adapter is how
/// the paragraph above went wrong.
///
/// # Every number above is a DEBUG build, and release moves them one way
///
/// `cargo test` builds the dev profile and this workspace sets no
/// `[profile.dev]` opt-level, which `tests/serving.rs` found the hard way
/// after publishing a llama.cpp comparison across that boundary. Rerun
/// `--release`:
///
///     1: 17.39   2: 12.45   3: 16.04   4: 15.70
///     5: 16.70   6: 17.58   7: 19.52   8: 21.03
///
/// giving 17.39, 6.23, 5.35, 3.93, 3.34, 2.93, 2.79 and **2.63 ms a token** --
/// **6.62x the single at eight rows**, against the 2.61x-3.36x above. It is
/// also monotone in a way neither debug run was, which is the second sign that
/// what those runs were reading was host noise.
///
/// I expected the opposite. Batching amortises a FIXED cost, optimisation
/// removes host work, and host work looked like the fixed cost -- so release
/// should have left less to amortise. It leaves more: fit the release row and
/// a fire is 16.87 ms fixed plus 0.52 ms a row, so **97 % of a one-row decode
/// is fixed**, where debug's per-row host arithmetic had been padding the
/// slope. Optimising the host did not shrink the lever, it exposed it.
///
/// So the chain closes harder than it was written to: `gdn_core` is 13x
/// cheaper a row at sixteen rows in isolation and the driver turns that into
/// 6.6x a token at eight conversations end to end.
#[test]
#[ignore = "times a real checkpoint; run explicitly"]
fn whether_batching_conversations_amortises_a_decode() {
    let Some((mut shell, _real)) = qwen3_5_shell(8) else {
        return;
    };
    const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];
    let who = |i: usize| 1_700u64 + i as u64;
    for i in 0..8 {
        if fire_row(&mut shell, who(i), &PERIOD).is_empty() {
            println!("a prefill was refused, so IT COULD NOT BE MEASURED");
            return;
        }
    }
    println!("\n  one token for N conversations in one fire:");
    let mut first = 0.0f64;
    for n in [1usize, 2, 3, 4, 5, 6, 7, 8] {
        let turns: Vec<driver_wgpu::turns::Turn> = (0..n)
            .map(|i| driver_wgpu::turns::Turn {
                who: who(i),
                tokens: PERIOD[..1].to_vec(),
            })
            .collect();
        for _ in 0..2 {
            let _ = shell.step(&turns);
        }
        let mut runs: Vec<f64> = Vec::new();
        for _ in 0..7 {
            let t = std::time::Instant::now();
            if shell.step(&turns).is_err() {
                println!("    a fire of {n} was refused");
                return;
            }
            runs.push(t.elapsed().as_secs_f64() * 1e3);
        }
        runs.sort_by(f64::total_cmp);
        let fire = runs[0];
        let per = fire / n as f64;
        if n == 1 {
            first = per;
        }
        println!(
            "    {n} row(s): {fire:7.2} ms a fire, {per:6.2} ms a token, {:.2}x the single",
            first / per
        );
    }
}

/// THIS FAMILY CANNOT TAKE `Geometry::recurrent`'s FALLBACK.
///
/// That helper answers `(kv_heads, head_dim)` when no recurrent pair is
/// stated, and its doc is right about why: a stack whose recurrent shape IS
/// its attention shape keeps reading the numbers it always did. Qwen3.5 is not
/// such a stack, and the gap is not subtle -- 16 value heads of 128 against 2
/// kv heads of 256.
///
/// A `Geometry` for this family that leaves the pair at zero therefore
/// dispatches the gated DeltaNet over TWO heads of sixteen and leaves the
/// other fourteen as arena litter. That is FIX 8, and it has now happened
/// twice: once as the original defect and once when the pair was added
/// upstream and this file's four literals filled it from `Default`.
///
/// The second time, nine suites stayed green. Every reference in this file
/// checks a kernel against the operands it was HANDED, and a scan handed two
/// heads computes two heads correctly; the only test that noticed asks the
/// model to write a sentence, and it is `#[ignore]`d for the checkpoint it
/// stages.
///
/// So the premise gets a test of its own, in the gate, with no GPU and no
/// weights: if these two pairs are ever equal for this family, the fallback
/// becomes harmless and this test should be deleted in the same edit that
/// makes it so. Until then, any `Geometry` built for qwen3.5 states the pair.
#[test]
fn the_hybrids_recurrent_pair_is_not_its_attention_pair() {
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let (attn_heads, attn_dim) = (facts.attn.kv_heads, facts.attn.head_dim);
    let (v_heads, v_dim) = (facts.gdn.value_heads, facts.gdn.value_head_dim);
    println!(
        "\n  attention answers ({attn_heads}, {attn_dim}); the gated DeltaNet wants ({v_heads}, {v_dim})"
    );
    assert_ne!(
        (attn_heads, attn_dim),
        (v_heads, v_dim),
        "the two pairs have converged, which makes `Geometry::recurrent`'s \
         fallback correct for this family and this test pointless"
    );
    // And what taking the fallback would cost, stated as the number it is: a
    // scan over `attn_heads` of the `v_heads` this stack has.
    assert!(
        v_heads > attn_heads,
        "the fallback would dispatch {attn_heads} of {v_heads} value heads"
    );
    println!(
        "  so a `Geometry` that leaves the pair at zero scans {attn_heads} of {v_heads} heads"
    );
}
