use std::collections::BTreeSet;

use crate::mini_dit_dims;

use mini_dit_dims as model;
use poem::{
    Attn, Def, Dim, Dtype, Elemwise, GeomKind, Guard, Operands, Operation, Platform, Request,
    RopeForm, RuntimeInput, Trace, Ty, ValueId,
};
use poem_ir::Conv;

const DEPLOYMENT: &str = "mini-dit-bf16-kv-bf16";

const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

fn trace(platform: Platform) -> Trace {
    let row = poem_compiler::catalog::deployment(DEPLOYMENT).unwrap_or_else(|| {
        let names: Vec<&str> = poem_compiler::catalog::deployments()
            .map(|row| row.name.as_str())
            .collect();
        panic!("this build ships no `{DEPLOYMENT}`; rows are {names:#?}")
    });
    row.trace(platform)
}

#[test]
fn the_mini_dit_row_bakes_every_case() {
    the_row_traces_on_every_platform_with_no_cache_and_a_velocity_readout();
    the_row_reads_exactly_the_five_inputs_it_names();
    each_stream_classifies_into_its_own_class_and_every_merge_resolves();
    the_joint_attention_is_self_paired_and_the_cross_attention_is_not();
    every_rope_turns_three_axes_of_the_whole_head_interleaved();
    the_modulation_is_a_per_lane_f32_pair_over_a_bf16_trunk();
    the_generative_facts_state_the_reading_its_latent_and_its_schedule();
    the_latent_is_patched_in_the_graph_and_the_positions_are_its_cells();
}

fn the_latent_is_patched_in_the_graph_and_the_positions_are_its_cells() {
    let plan = trace(Platform::Cuda);
    let patch = [1, model::PATCH, model::PATCH];
    let ops: Vec<&Conv> = plan
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Conv(op) => Some(op),
            _ => None,
        })
        .collect();
    let patched = ops
        .iter()
        .filter(|op| matches!(op, Conv::Patchify { p, .. } if *p == patch))
        .count();
    let unpatched = ops
        .iter()
        .filter(|op| matches!(op, Conv::Unpatchify { p, .. } if *p == patch))
        .count();
    assert_eq!(
        (patched, unpatched),
        (1, 1),
        "the latent in, the velocity out"
    );
    let cells: Vec<u32> = ops
        .iter()
        .filter_map(|op| match op {
            Conv::Cells { axis, .. } => Some(*axis),
            _ => None,
        })
        .collect();
    assert_eq!(
        cells,
        vec![0],
        "a text row's position runs down the time axis"
    );

    let velocity = plan
        .outputs
        .iter()
        .find(|s| s.name == "velocity")
        .expect("a velocity seam")
        .value;
    assert_eq!(
        plan.values[velocity.0 as usize].ty,
        Ty::Tensor {
            shape: vec![Dim::Voxels, Dim::Const(u64::from(model::CHANNELS))],
            dtype: Dtype::Bf16,
        },
        "the velocity reads out in the latent's own voxels"
    );
    assert_eq!(
        plan.facts.declared("positions").map(|s| s.values.clone()),
        Some(vec!["computed".to_string(), "given".to_string()])
    );

    let budgets = poem_compiler::Budgets::of(poem_compiler::Budget::new(4, 256)).with_voxels(
        poem_compiler::VoxelLadder {
            max_voxels: 1024,
            buckets: vec![256, 1024],
            max_clips: 4,
        },
    );
    let compiled =
        poem_compiler::compile_axes(&plan, &budgets, &poem_compiler::DeviceProfile::default())
            .expect("the voxel-in, voxel-out denoiser bakes");
    assert!(
        compiled.units.contains(&poem_compiler::RowAxis::Voxels)
            && compiled.units.contains(&poem_compiler::RowAxis::Tokens),
        "the latent's voxels and the tokens are two units: {:?}",
        compiled.units
    );
    assert!(compiled.arena.clashes(&compiled.concurrency).is_empty());
}

fn the_row_traces_on_every_platform_with_no_cache_and_a_velocity_readout() {
    for platform in PLATFORMS {
        let plan = trace(platform);
        assert!(!plan.nodes.is_empty(), "{platform:?}: an empty plan");
        assert!(
            plan.caches.is_empty(),
            "{platform:?}: a denoise reading holds nothing between fires, and this row \
             declared {:?}",
            plan.caches,
        );
        let seams: BTreeSet<&str> = plan.outputs.iter().map(|s| s.name.as_str()).collect();
        assert!(
            seams.contains("velocity"),
            "{platform:?}: no velocity readout; seams are {seams:?}"
        );
        assert!(
            !seams.contains("logits"),
            "{platform:?}: a denoiser has no logits, and `out` was planted anyway"
        );
    }
}

fn the_row_reads_exactly_the_five_inputs_it_names() {
    let plan = trace(Platform::Cuda);
    let named: BTreeSet<String> = plan
        .values
        .iter()
        .filter_map(|decl| match &decl.def {
            Def::Input(RuntimeInput::Guest {
                id, axis, width, ..
            }) => Some(format!(
                "{}:{}:{width}",
                plan.inputs[*id as usize].name,
                axis.name()
            )),
            _ => None,
        })
        .collect();
    let want: BTreeSet<String> = [
        format!("latents:voxel:{}", model::CHANNELS),
        format!("text:row:{}", model::TEXT_WIDTH),
        format!("context:row:{}", model::CONTEXT_WIDTH),
        "timestep:lane:1".to_string(),
        format!("positions:row:{}", model::ROPE_AXES),
    ]
    .into_iter()
    .collect();
    assert_eq!(named, want);

    let sinusoids: Vec<(u32, f32, bool, f32)> = plan
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Elemwise(Elemwise::Sinusoid {
                dim,
                max_period,
                flip_sin_cos,
                scale,
                ..
            }) => Some((*dim, *max_period, *flip_sin_cos, *scale)),
            _ => None,
        })
        .collect();
    assert_eq!(
        sinusoids,
        vec![(
            model::TIMESTEP_DIM,
            model::TIMESTEP_MAX_PERIOD,
            model::TIMESTEP_FLIP_SIN_COS,
            model::TIMESTEP_SCALE
        )]
    );
}

fn each_stream_classifies_into_its_own_class_and_every_merge_resolves() {
    let plan = trace(Platform::Cuda);
    let classes = poem::resolve_classes(&plan).expect("every merge resolves");
    let _row = poem_compiler::catalog::deployment(DEPLOYMENT).expect("the row is in the catalog");

    let mut seen = Vec::new();
    for stream in ["text", "image", "context"] {
        let request = Request::new(1, false).switched("stream", stream);
        let word = plan.facts.word(&request);
        let class = classes
            .class_of(word & classes.mask)
            .unwrap_or_else(|| panic!("a {stream:?} lane has no class"));
        seen.push((stream, class));
    }
    let distinct: BTreeSet<usize> = seen.iter().map(|(_, class)| *class).collect();
    assert_eq!(
        distinct.len(),
        3,
        "the three streams share a class: {seen:?}"
    );
}

fn the_joint_attention_is_self_paired_and_the_cross_attention_is_not() {
    let plan = trace(Platform::Cuda);
    let ragged: Vec<(ValueId, ValueId, u32)> = plan
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Attn(Attn::Ragged {
                q_indptr,
                kv_indptr,
                head_dim,
                ..
            }) => Some((*q_indptr, *kv_indptr, *head_dim)),
            _ => None,
        })
        .collect();
    assert_eq!(ragged.len(), 4, "one ragged read per attention sublayer");
    assert!(
        ragged
            .iter()
            .all(|(_, _, head_dim)| *head_dim == model::HEAD_DIM),
        "every head is {} wide",
        model::HEAD_DIM
    );
    let self_paired: Vec<&(ValueId, ValueId, u32)> =
        ragged.iter().filter(|(q, kv, _)| q == kv).collect();
    let crossed: Vec<&(ValueId, ValueId, u32)> =
        ragged.iter().filter(|(q, kv, _)| q != kv).collect();
    assert_eq!(self_paired.len(), 3);
    assert_eq!(
        crossed.len(),
        1,
        "one attention whose keys are another lane's"
    );

    let selection_of = |id: ValueId| match &plan.values[id.0 as usize].def {
        Def::Input(RuntimeInput::Geometry {
            kind: GeomKind::GroupIndptr { select },
            ..
        }) => *select,
        other => panic!("a ragged CSR that is not a group indptr: {other:?}"),
    };
    let joint = selection_of(self_paired[0].0);
    assert_eq!(
        selection_of(self_paired[1].0),
        joint,
        "blocks 0 and 1 pack one sequence"
    );
    let word = |stream: &str| {
        plan.facts
            .word(&Request::new(1, false).switched("stream", stream))
    };
    assert!(joint.holds(word("text")));
    assert!(joint.holds(word("image")));
    assert!(!joint.holds(word("context")));

    let (q_csr, kv_csr, _) = crossed[0];
    assert!(selection_of(*q_csr).holds(word("image")));
    assert!(!selection_of(*q_csr).holds(word("context")));
    assert!(selection_of(*kv_csr).holds(word("context")));
    assert!(!selection_of(*kv_csr).holds(word("image")));

    let joins = plan
        .nodes
        .iter()
        .find(|node| {
            matches!(
                &node.op,
                Operation::Attn(Attn::Ragged { q_indptr, kv_indptr, .. })
                    if q_indptr != kv_indptr
            )
        })
        .expect("the cross attention");
    assert!(
        matches!(joins.guard, Guard::Or(..)),
        "a cross attention spans both arms: {:?}",
        joins.guard
    );
}

fn every_rope_turns_three_axes_of_the_whole_head_interleaved() {
    let plan = trace(Platform::Cuda);
    let ropes: Vec<([u32; 4], RopeForm, u32, u32)> = plan
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Elemwise(Elemwise::RopeAxes {
                dims,
                form,
                rotary_dim,
                head_dim,
                ..
            }) => Some((*dims, *form, *rotary_dim, *head_dim)),
            _ => None,
        })
        .collect();
    assert_eq!(ropes.len(), 8);
    for rope in &ropes {
        assert_eq!(
            *rope,
            (
                model::ROPE_DIMS,
                RopeForm::Interleaved,
                model::HEAD_DIM,
                model::HEAD_DIM
            )
        );
    }
    assert_eq!(
        model::ROPE_DIMS.iter().sum::<u32>(),
        model::HEAD_DIM,
        "the three axes cover the whole head"
    );

    let ropes_after_cross = plan
        .nodes
        .iter()
        .skip_while(|node| {
            !matches!(
                &node.op,
                Operation::Elemwise(Elemwise::Layernorm {
                    weight: Some(_),
                    bias: Some(_),
                    ..
                })
            )
        })
        .filter(|node| matches!(&node.op, Operation::Elemwise(Elemwise::RopeAxes { .. })))
        .count();
    assert_eq!(
        ropes_after_cross, 0,
        "the affine cross norm is followed by no rope"
    );
}

fn the_modulation_is_a_per_lane_f32_pair_over_a_bf16_trunk() {
    let plan = trace(Platform::Cuda);
    let ty = |id: ValueId| plan.values[id.0 as usize].ty.clone();
    let mut seen = 0usize;
    for node in &plan.nodes {
        let Operation::Elemwise(Elemwise::Modulate {
            x, m, lane_of_row, ..
        }) = &node.op
        else {
            continue;
        };
        seen += 1;
        assert_eq!(
            ty(*x),
            Ty::Tensor {
                shape: vec![Dim::Tokens, Dim::Const(u64::from(model::HIDDEN))],
                dtype: Dtype::Bf16,
            }
        );
        assert_eq!(
            ty(*m),
            Ty::Tensor {
                shape: vec![Dim::Lanes, Dim::Const(u64::from(2 * model::HIDDEN))],
                dtype: Dtype::F32,
            }
        );
        let lanes = lane_of_row.expect("every modulation here is per lane");
        assert_eq!(
            plan.values[lanes.0 as usize].def,
            Def::Input(RuntimeInput::Geometry {
                space: 0,
                kind: GeomKind::RequestOfToken
            }),
            "the broadcast is the fire's token→lane table"
        );
    }
    assert_eq!(seen, 9);

    for node in &plan.nodes {
        if !matches!(
            &node.op,
            Operation::Elemwise(Elemwise::ResidualAdd { gate: Some(_), .. })
        ) {
            continue;
        }
        let mut pairs = Vec::new();
        node.op.aliases(&mut pairs);
        assert_eq!(pairs.len(), 1, "a gated fold is in place on its residual");
    }
}

fn the_generative_facts_state_the_reading_its_latent_and_its_schedule() {
    let row = poem_compiler::catalog::deployment(DEPLOYMENT).expect("the row is in the catalog");
    let facts = row
        .generative()
        .expect("the first generative row states its readings");
    let declared = |name: &str| {
        trace(Platform::Cuda)
            .facts
            .declared(name)
            .map(|s| s.values.clone())
    };
    assert_eq!(
        declared("reading"),
        Some(vec!["denoise".to_string()]),
        "one reading, today"
    );
    assert_eq!(
        declared("stream"),
        Some(["text", "image", "context"].map(String::from).to_vec())
    );

    let latent = facts.latent.expect("a denoiser states its latent space");
    assert_eq!(latent.channels, model::CHANNELS);
    let schedule = facts.schedule.as_ref().expect("and its schedule");
    assert_eq!(schedule.kind, poem::generative::ScheduleKind::Flow);
    assert_eq!(schedule.pinned_sigmas, vec![1.0, 0.75, 0.5, 0.25]);
}
