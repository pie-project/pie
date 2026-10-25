use std::collections::{BTreeMap, BTreeSet};

use poem::generative::ScheduleKind;
use poem::{
    Attn, Def, Dim, Dtype, Elemwise, Operation, Platform, RaggedMask, Request, RopeForm,
    RuntimeInput, Trace, Ty, seam,
};
use poem_ir::InputAxis;

use crate::minimax_h3_dims::{self as model, Dims};

const FLAGSHIP: &str = "minimax-h3-fl2va-bf16-kv-bf16";
const MINI: &str = "minimax-h3-mini-bf16-kv-bf16";
const ROWS: [&str; 2] = [FLAGSHIP, MINI];

const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

fn row(deployment: &str) -> &'static poem_compiler::catalog::Deployment {
    poem_compiler::catalog::deployment(deployment).unwrap_or_else(|| {
        let names: Vec<&str> = poem_compiler::catalog::deployments()
            .map(|row| row.name.as_str())
            .collect();
        panic!("this build ships no `{deployment}`; rows are {names:#?}")
    })
}

fn trace(deployment: &str, platform: Platform) -> Trace {
    row(deployment).trace(platform)
}

fn dims(deployment: &str) -> Dims {
    match deployment {
        FLAGSHIP => Dims::h3(),
        MINI => Dims::mini(),
        other => panic!("no dims for `{other}`"),
    }
}

fn is_flagship(deployment: &str) -> bool {
    deployment == FLAGSHIP
}

fn seams(plan: &Trace) -> BTreeMap<&str, usize> {
    plan.seams.iter().fold(BTreeMap::new(), |mut acc, s| {
        *acc.entry(s.seam.as_str()).or_default() += 1;
        acc
    })
}

#[test]
fn the_minimax_h3_rows_bake_every_case() {
    the_video_and_reference_latents_are_patched_in_the_graph();
    every_row_traces_on_every_platform_with_the_encoder_it_declares();
    the_seams_are_the_two_float_readouts_and_never_logits();
    every_lane_the_facts_list_lands_in_a_class_where_the_merges_resolve();
    one_joint_read_per_block_and_one_lane_read_per_refiner();
    the_trunk_turns_three_neox_axes_and_the_refiner_turns_nothing();
    the_modality_gather_is_three_weight_blocks_and_one_column_slice();
    the_sharded_worlds_trace_and_bake();
    the_generative_facts_state_the_readings_the_latent_and_two_shifts();
    the_modulation_is_a_lane_vector_and_every_gated_fold_folds_its_residual();
    the_gather_merges_four_lane_shaped_arms_back_onto_the_reading();
    the_positions_are_computed_in_the_graph_unless_given();
}

fn the_positions_are_computed_in_the_graph_unless_given() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let count = |f: &dyn Fn(&Operation) -> bool| plan.nodes.iter().filter(|n| f(&n.op)).count();
        assert_eq!(
            count(&|op| matches!(op, Operation::Conv(poem_ir::Conv::Cells { axis: 0, .. }))),
            2,
            "`{deployment}`: every row's cell, and the video lane's own for its peers"
        );
        assert_eq!(
            count(&|op| matches!(op, Operation::Gather(poem_ir::Gather::GroupFirst { .. }))),
            1,
            "`{deployment}`: the audio pins to its group's video grid"
        );
        assert_eq!(
            count(&|op| matches!(op, Operation::Gather(poem_ir::Gather::GroupRows { .. }))),
            2,
            "`{deployment}`: the text prefix and the audio ticks, read off the group"
        );
        assert_eq!(
            plan.facts
                .declared("positions")
                .map(|s| s.values.clone())
                .unwrap_or_default(),
            vec!["computed", "given"],
            "`{deployment}`: positions are computed unless a lane gives them"
        );
        let positions = plan
            .inputs
            .iter()
            .position(|i| i.name == "positions")
            .expect("a positions input");
        assert!(
            plan.values.iter().any(|v| matches!(v.def,
                Def::Input(RuntimeInput::Guest { id, axis: InputAxis::Row, width, .. })
                    if id as usize == positions && width == 3)),
            "`{deployment}`: a `given` lane reads (t, h, w) a row"
        );
    }
}

fn every_row_traces_on_every_platform_with_the_encoder_it_declares() {
    for platform in PLATFORMS {
        for deployment in ROWS {
            let plan = trace(deployment, platform);
            assert!(
                !plan.nodes.is_empty(),
                "{platform:?}: `{deployment}` traced no node"
            );
            let kv_rows = plan.caches.len();
            let wanted = if is_flagship(deployment) {
                model::TE_LAYERS as usize
            } else {
                0
            };
            assert_eq!(
                kv_rows, wanted,
                "{platform:?} `{deployment}`: one kv row per encoder layer the plan runs"
            );
        }
    }
}

fn the_seams_are_the_two_float_readouts_and_never_logits() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let found = seams(&plan);
        assert!(
            !found.contains_key(seam::OUT.name),
            "`{deployment}`: no reading of this family has logits, and it planted `{}`",
            seam::OUT.name
        );
        assert_eq!(
            found.get(seam::VELOCITY.name),
            Some(&1),
            "`{deployment}`: one velocity planting (the video head); found {found:?}"
        );
        assert_eq!(
            found.get(seam::HIDDEN.name),
            Some(&if is_flagship(deployment) { 3 } else { 2 }),
            "`{deployment}`: hidden plantings; found {found:?}"
        );
    }
}

/// Every (reading, stream) a lane of `deployment` may take.
fn lanes(deployment: &str) -> Vec<(&'static str, &'static str)> {
    let mut lanes = Vec::new();
    if is_flagship(deployment) {
        lanes.push(("text", "text"));
    }
    lanes.push(("refine", "text"));
    for stream in ["text", "video", "audio", "reference"] {
        lanes.push(("denoise", stream));
    }
    lanes
}

fn every_lane_the_facts_list_lands_in_a_class_where_the_merges_resolve() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let classes = poem::resolve_classes(&plan).expect("every merge resolves");
        let _catalog = row(deployment);
        let mut seen: Vec<((&str, &str), usize)> = Vec::new();
        for (name, stream) in lanes(deployment) {
            let request = Request::new(4, false)
                .switched("stream", stream)
                .switched("reading", name);
            let word = plan.facts.word(&request);
            let class = classes
                .class_of(word & classes.mask)
                .unwrap_or_else(|| panic!("`{deployment}`: a {name}/{stream:?} lane has no class"));
            seen.push(((name, stream), class));
        }
        let distinct: BTreeSet<usize> = seen.iter().map(|(_, class)| *class).collect();
        assert_eq!(
            distinct.len(),
            seen.len(),
            "`{deployment}`: two lanes share a class: {seen:?}"
        );
    }
}

fn ragged(plan: &Trace) -> Vec<RaggedMask> {
    plan.nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Attn(Attn::Ragged { mask, .. }) => Some(*mask),
            _ => None,
        })
        .collect()
}

fn one_joint_read_per_block_and_one_lane_read_per_refiner() {
    for deployment in ROWS {
        let d = dims(deployment);
        let plan = trace(deployment, Platform::Cuda);
        let reads = ragged(&plan);
        let joint = reads
            .iter()
            .filter(|mask| matches!(mask, RaggedMask::GroupBlockDiagonal))
            .count();
        let lane = reads
            .iter()
            .filter(|mask| matches!(mask, RaggedMask::None))
            .count();
        assert_eq!(
            joint, d.blocks as usize,
            "`{deployment}`: one joint attention per trunk block"
        );
        assert_eq!(
            lane, d.refiners as usize,
            "`{deployment}`: one lane-local attention per refiner block"
        );
        let prefill = plan
            .nodes
            .iter()
            .filter(|node| {
                matches!(
                    &node.op,
                    Operation::Attn(Attn::Prefill {
                        mask: None,
                        selection: None,
                        bias: None,
                        lse: None,
                        ..
                    })
                )
            })
            .count();
        assert_eq!(
            prefill,
            if is_flagship(deployment) {
                model::TE_LAYERS as usize
            } else {
                0
            },
            "`{deployment}`: the encoder's paged prefill, one per layer it runs"
        );
    }
}

fn the_trunk_turns_three_neox_axes_and_the_refiner_turns_nothing() {
    for deployment in ROWS {
        let d = dims(deployment);
        let plan = trace(deployment, Platform::Cuda);
        let mut turns = 0usize;
        for node in &plan.nodes {
            let Operation::Elemwise(Elemwise::RopeAxes {
                dims,
                thetas,
                form,
                rotary_dim,
                head_dim,
                ..
            }) = &node.op
            else {
                continue;
            };
            turns += 1;
            assert_eq!(
                *dims,
                d.rope_dims(),
                "`{deployment}`: the three axes' channels"
            );
            assert_eq!(
                thetas[..3],
                [model::ROPE_THETA; 3],
                "`{deployment}`: one base for every axis"
            );
            assert_eq!(*form, RopeForm::Neox, "`{deployment}`: rotate-half pairing");
            assert_eq!(
                *rotary_dim,
                d.rotary_dim(),
                "`{deployment}`: the rotated span"
            );
            assert_eq!(*head_dim, d.head_dim, "`{deployment}`: the head width");
        }
        assert_eq!(
            turns,
            2 * d.blocks as usize,
            "`{deployment}`: only the trunk's blocks turn"
        );
    }
}

fn the_modality_gather_is_three_weight_blocks_and_one_column_slice() {
    for deployment in ROWS {
        let d = dims(deployment);
        let plan = trace(deployment, Platform::Cuda);
        for i in 0..d.blocks {
            for which in 0..model::MODALITIES {
                let name = format!("dit.blocks.{i}.adaln.{which}.w");
                let bank = plan
                    .params
                    .iter()
                    .find(|p| p.name == name)
                    .unwrap_or_else(|| panic!("`{deployment}` block {i}: no `{name}`"));
                assert_eq!(
                    bank.shape,
                    vec![u64::from(d.adaln_width()), u64::from(d.t_dim)],
                    "`{deployment}` block {i} modality {which}: `[6·dim, t_dim]`"
                );
            }
        }
        let pairs: BTreeSet<(u32, usize)> = ["text", "video", "audio", "reference"]
            .into_iter()
            .map(|s| (model::timestep_slot(s), model::modality(s)))
            .collect();
        assert_eq!(
            pairs.len(),
            4,
            "`{deployment}`: the four lanes gather four different modulation vectors"
        );
        let timestep = plan
            .inputs
            .iter()
            .position(|input| input.name == "timestep")
            .expect("a timestep input") as u32;
        let widths: BTreeSet<u32> = plan
            .values
            .iter()
            .filter_map(|decl| match &decl.def {
                Def::Input(RuntimeInput::Guest {
                    id,
                    axis: InputAxis::Lane,
                    width,
                    ..
                }) if *id == timestep => Some(*width),
                _ => None,
            })
            .collect();
        assert_eq!(
            widths,
            BTreeSet::from([model::TIMESTEP_SLOTS]),
            "`{deployment}`: the timestep input carries every unique timestep of the step"
        );
    }
}

fn budget() -> poem_compiler::Budget {
    poem_compiler::Budget {
        max_lanes: 64,
        max_tokens: 4096,
        buckets: vec![64, 256, 1024, 4096],
        max_adapters: 0,
        readouts_per_lane: poem_compiler::arena::READOUTS_PER_LANE,
    }
}

fn the_sharded_worlds_trace_and_bake() {
    for tp in [2u32, 4] {
        let deployment = format!("minimax-h3-fl2va-bf16-kv-bf16-tp{tp}");
        let d = Dims::h3();
        let plan = trace(&deployment, Platform::Cuda);
        assert_eq!(
            plan.caches.len(),
            model::TE_LAYERS as usize,
            "tp {tp}: one kv row per encoder layer"
        );
        let budgets =
            poem_compiler::Budgets::of(budget()).with_voxels(poem_compiler::VoxelLadder {
                max_voxels: 16_384,
                buckets: vec![1024, 16_384],
                max_clips: 8,
            });
        poem_compiler::compile_axes(&plan, &budgets, &poem_compiler::DeviceProfile::default())
            .unwrap_or_else(|why| panic!("`{deployment}` does not bake: {why}"));
        for which in 0..model::MODALITIES {
            let name = format!("dit.blocks.0.adaln.{which}.w");
            let bank = plan
                .params
                .iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("tp {tp}: no `{name}`"));
            assert_eq!(
                bank.shape,
                vec![u64::from(Dims::h3().adaln_width()), u64::from(d.t_dim)],
                "tp {tp}: the modulation bank is replicated, not cut"
            );
        }
    }
}

fn the_generative_facts_state_the_readings_the_latent_and_two_shifts() {
    for deployment in ROWS {
        let facts = row(deployment).generative().expect("facts");
        let names = trace(deployment, Platform::Cuda)
            .facts
            .declared("reading")
            .map(|s| s.values.clone())
            .unwrap_or_default();
        assert_eq!(
            names,
            if is_flagship(deployment) {
                vec!["text", "refine", "denoise"]
            } else {
                vec!["refine", "denoise"]
            },
            "`{deployment}`: the readings a guest may name"
        );
        let latent = facts.latent.expect("a latent space");
        assert_eq!(
            (
                latent.channels,
                latent.spatial_compression,
                latent.temporal_compression
            ),
            (24, 16, 4),
            "`{deployment}`: the f16 t4 c24 latent"
        );
        let schedule = facts.schedule.as_ref().expect("a schedule");
        assert_eq!(schedule.kind, ScheduleKind::Flow);
        assert_eq!(schedule.shift, model::VIDEO_SHIFT);
        assert_eq!(
            schedule.stream_shifts.iter().find(|(s, _)| s == "video"),
            Some(&("video".to_string(), model::VIDEO_SHIFT)),
            "`{deployment}`: the video grid's shift"
        );
        assert_eq!(
            schedule.stream_shifts.iter().find(|(s, _)| s == "audio"),
            Some(&("audio".to_string(), model::AUDIO_SHIFT)),
            "`{deployment}`: the audio grid's shift, run in the same evaluation"
        );
        assert_eq!(
            schedule.pinned_sigmas.len(),
            model::STEPS as usize - 1,
            "`{deployment}`: fifty grid points make forty-nine evaluations"
        );
        assert!(
            schedule.pinned_sigmas.windows(2).all(|w| w[0] > w[1]),
            "`{deployment}`: the sigma grid descends"
        );
    }
}

fn the_modulation_is_a_lane_vector_and_every_gated_fold_folds_its_residual() {
    for deployment in ROWS {
        let d = dims(deployment);
        let plan = trace(deployment, Platform::Cuda);
        let mut modulates = 0usize;
        let mut folds = 0usize;
        for node in &plan.nodes {
            match &node.op {
                Operation::Elemwise(Elemwise::Modulate { m, lane_of_row, .. }) => {
                    modulates += 1;
                    assert!(
                        lane_of_row.is_some(),
                        "`{deployment}`: H3 modulates per LANE, broadcast by request_of_token"
                    );
                    let ty = &plan.values[m.0 as usize].ty;
                    assert!(
                        matches!(ty, Ty::Tensor { shape, dtype: Dtype::F32 }
                            if matches!(shape.as_slice(), [Dim::Lanes, _])),
                        "`{deployment}`: a lane-vector chain lands `[Lanes, ·]` f32, not {ty:?}"
                    );
                }
                Operation::Elemwise(Elemwise::ResidualAdd {
                    gate: Some(_),
                    y: r,
                    y_out: r_out,
                    ..
                }) => {
                    folds += 1;
                    assert_eq!(
                        plan.values[r.0 as usize].ty, plan.values[r_out.0 as usize].ty,
                        "`{deployment}`: a gated fold answers its residual's own rectangle"
                    );
                }
                _ => {}
            }
        }
        assert_eq!(
            modulates,
            2 * d.blocks as usize + 1,
            "`{deployment}`: two modulated sublayers a block and one at the head"
        );
        assert_eq!(
            folds,
            2 * d.blocks as usize,
            "`{deployment}`: two gated folds a block"
        );
    }
}

fn the_gather_merges_four_lane_shaped_arms_back_onto_the_reading() {
    for deployment in ROWS {
        let d = dims(deployment);
        let plan = trace(deployment, Platform::Cuda);
        let lane_merges = plan
            .values
            .iter()
            .filter(|value| match (&value.def, &value.ty) {
                (Def::Merge(arms), Ty::Tensor { shape, .. }) => {
                    arms.len() == 4 && matches!(shape.as_slice(), [Dim::Lanes, _])
                }
                _ => false,
            })
            .count();
        assert_eq!(
            lane_merges,
            d.blocks as usize + 1,
            "`{deployment}`: the four sides' modulation vectors come back as one rectangle"
        );
        let token_merges = plan
            .values
            .iter()
            .filter(|value| match (&value.def, &value.ty) {
                (Def::Merge(arms), Ty::Tensor { shape, .. }) => {
                    arms.len() == 4 && matches!(shape.as_slice(), [Dim::Tokens, _])
                }
                _ => false,
            })
            .count();
        assert_eq!(
            token_merges, 2,
            "`{deployment}`: the packed row sequence and its computed positions, each joined once"
        );
    }
}

fn the_video_and_reference_latents_are_patched_in_the_graph() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let voxel_inputs: BTreeSet<(String, u32)> = plan
            .values
            .iter()
            .filter_map(|decl| match &decl.def {
                Def::Input(RuntimeInput::Guest {
                    id,
                    axis: InputAxis::Voxel,
                    width,
                    ..
                }) => Some((plan.inputs[*id as usize].name.clone(), *width)),
                _ => None,
            })
            .collect();
        assert_eq!(
            voxel_inputs,
            BTreeSet::from([
                ("latents".to_string(), model::LATENT_CHANNELS),
                ("reference".to_string(), model::LATENT_CHANNELS),
            ]),
            "`{deployment}`: the video and reference clips share the voxel axis at one width"
        );
        let patch = [1, 2, 2];
        let count = |f: &dyn Fn(&poem_ir::Conv) -> bool| {
            plan.nodes
                .iter()
                .filter(|node| matches!(&node.op, Operation::Conv(op) if f(op)))
                .count()
        };
        assert_eq!(
            count(&|op| matches!(op, poem_ir::Conv::Patchify { p, .. } if *p == patch)),
            2,
            "`{deployment}`: the target and the reference patched in"
        );
        assert_eq!(
            count(&|op| matches!(op, poem_ir::Conv::Unpatchify { p, .. } if *p == patch)),
            1,
            "`{deployment}`: the video velocity patched back out"
        );
        let velocity = plan
            .seams
            .iter()
            .find(|s| s.seam == seam::VELOCITY.name)
            .expect("a velocity seam")
            .values[0];
        assert!(
            matches!(&plan.values[velocity.0 as usize].ty,
                Ty::Tensor { shape, .. } if shape.as_slice()
                    == [Dim::Voxels, Dim::Const(u64::from(model::LATENT_CHANNELS))]),
            "`{deployment}`: the velocity reads out in the latent's own voxels"
        );
    }
}
