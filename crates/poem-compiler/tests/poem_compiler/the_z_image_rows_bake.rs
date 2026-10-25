use std::collections::{BTreeMap, BTreeSet};

use crate::z_image_dims::{self as model, Dims};
use poem::generative::ScheduleKind;
use poem::{
    Attn, Def, Dim, Dtype, Elemwise, Gather, GeomKind, InputAxis, Operation, Platform, Request,
    RopeForm, RuntimeInput, Selection, Trace, Ty, ValueId, seam,
};
use poem_ir::Conv;

type RopeRow = ([u32; 4], [f32; 4], RopeForm, u32, u32);

const TURBO: &str = "z-image-turbo-bf16-kv-bf16";
const MINI: &str = "z-image-mini-bf16-kv-bf16";
const ROWS: [&str; 2] = [TURBO, MINI];

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
    if deployment == TURBO {
        Dims::turbo()
    } else {
        Dims::mini()
    }
}

/// Every (reading, stream) a lane of `deployment` may take, its stream the
/// declared default where none is given.
fn lanes(deployment: &str) -> Vec<(&'static str, Option<&'static str>)> {
    let mut lanes = vec![
        ("refine", Some("context")),
        ("denoise", Some("image")),
        ("denoise", Some("context")),
    ];
    if deployment == TURBO {
        lanes.extend([("text", None), ("vae.decode", None), ("vae.encode", None)]);
    }
    lanes
}

/// The word of a lane in `reading`, on `stream` where given (its declared
/// default otherwise).
fn word(plan: &Trace, reading: &str, stream: Option<&str>) -> u64 {
    let mut request = Request::new(4, false).switched("reading", reading);
    if let Some(stream) = stream {
        request = request.switched("stream", stream);
    }
    plan.facts.word(&request)
}

#[test]
fn the_z_image_rows_bake_every_case() {
    every_row_traces_on_every_platform_with_the_caches_and_readouts_it_states();
    the_readings_state_their_readouts_and_the_sinusoid();
    every_declared_lane_classifies_into_its_own_class_where_every_merge_resolves();
    the_refiners_attend_within_a_lane_and_the_trunk_within_the_group();
    the_dit_turns_three_interleaved_axes_and_the_encoder_the_whole_neox_head();
    the_modulation_is_a_per_lane_f32_scale_over_a_bf16_trunk();
    the_generative_facts_state_the_readings_the_schedule_and_the_latent_space();
    the_latent_is_patched_in_the_graph_and_the_positions_follow_the_caption();
}

fn the_latent_is_patched_in_the_graph_and_the_positions_follow_the_caption() {
    let patch = [1, model::PATCH, model::PATCH];
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let count = |f: &dyn Fn(&Operation) -> bool| plan.nodes.iter().filter(|n| f(&n.op)).count();
        assert_eq!(
            count(&|op| matches!(op, Operation::Conv(Conv::Patchify { p, .. }) if *p == patch)),
            1,
            "{deployment}: the voxel latent patches in the graph"
        );
        assert_eq!(
            count(&|op| matches!(op, Operation::Conv(Conv::Unpatchify { p, .. }) if *p == patch)),
            1,
            "{deployment}: the velocity reads out in the latent's voxels"
        );
        let cells: Vec<u32> = plan
            .nodes
            .iter()
            .filter_map(|n| match &n.op {
                Operation::Conv(Conv::Cells { axis, .. }) => Some(*axis),
                _ => None,
            })
            .collect();
        assert_eq!(
            cells,
            vec![0, 0, 0],
            "{deployment}: a caption row runs down time: the refine's, the context's and the image's"
        );
        assert_eq!(
            count(&|op| matches!(op, Operation::Gather(Gather::GroupRows { .. }))),
            1,
            "{deployment}: the image follows its group's caption rows"
        );
        for name in ["caption", "context", "latents"] {
            let input = plan
                .inputs
                .iter()
                .find(|i| i.name == name)
                .unwrap_or_else(|| panic!("{deployment}: no `{name}` input"));
            assert_eq!(
                input.rows,
                Some(poem_ir::RowCount::Multiple(32)),
                "{deployment}: `{name}`'s lane is padded to a multiple of 32 rows, as the reference pads"
            );
        }
        let length = plan
            .inputs
            .iter()
            .position(|i| i.name == "length")
            .expect("a length input");
        assert!(
            plan.values.iter().any(|v| matches!(v.def,
                Def::Input(RuntimeInput::Guest { id, axis: InputAxis::Lane, width: 1, .. })
                    if id as usize == length)),
            "{deployment}: a padded lane states its own rows, one value a lane"
        );
        let latents = plan
            .inputs
            .iter()
            .position(|i| i.name == "latents")
            .expect("a latents input");
        assert!(
            plan.values.iter().any(|v| matches!(v.def,
                Def::Input(RuntimeInput::Guest { id, axis: InputAxis::Voxel, width, .. })
                    if id as usize == latents && width == model::CHANNELS)),
            "{deployment}: the latent is VAE-native, one voxel of {} channels",
            model::CHANNELS
        );
        let velocity = plan
            .seams
            .iter()
            .find(|s| s.seam == seam::VELOCITY.name)
            .expect("a velocity seam")
            .values[0];
        assert_eq!(
            plan.values[velocity.0 as usize].ty,
            Ty::Tensor {
                shape: vec![Dim::Voxels, Dim::Const(u64::from(model::CHANNELS))],
                dtype: Dtype::Bf16,
            },
            "{deployment}"
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
                .unwrap_or_else(|why| panic!("{deployment} bakes: {why}"));
        assert!(
            compiled.units.contains(&poem_compiler::RowAxis::Voxels)
                && compiled.units.contains(&poem_compiler::RowAxis::Tokens),
            "{deployment}: {:?}",
            compiled.units
        );
        assert!(compiled.arena.clashes(&compiled.concurrency).is_empty());
    }
}

fn every_row_traces_on_every_platform_with_the_caches_and_readouts_it_states() {
    for deployment in ROWS {
        for platform in PLATFORMS {
            let plan = trace(deployment, platform);
            assert!(
                !plan.nodes.is_empty(),
                "{deployment} {platform:?}: an empty plan"
            );
            let want = if deployment == TURBO {
                model::TE_LAYERS as usize
            } else {
                0
            };
            assert_eq!(
                plan.caches.len(),
                want,
                "{deployment} {platform:?}: the encoder's kv rows, and nothing else, are held"
            );
            let seams: BTreeMap<&str, usize> =
                plan.seams.iter().fold(BTreeMap::new(), |mut acc, s| {
                    *acc.entry(s.seam.as_str()).or_default() += 1;
                    acc
                });
            assert_eq!(
                seams.get(seam::VELOCITY.name),
                Some(&1),
                "{deployment}: one velocity"
            );
            assert!(
                !seams.contains_key(seam::OUT.name),
                "{deployment}: a denoiser has no logits, and `out` was planted anyway"
            );
            let hidden = if deployment == TURBO { 2 } else { 1 };
            assert_eq!(
                seams.get(seam::HIDDEN.name),
                Some(&hidden),
                "{deployment}: one `hidden` per reading that reads hidden rows back"
            );
            if deployment == TURBO {
                let taps: BTreeSet<Option<u32>> = plan
                    .seams
                    .iter()
                    .filter(|s| s.seam == seam::HIDDEN.name)
                    .map(|s| s.layer)
                    .collect();
                assert!(
                    taps.contains(&Some(model::TE_LAYERS - 1)),
                    "the encoder tap is the residual leaving layer {}: {taps:?}",
                    model::TE_LAYERS - 1
                );
            }
        }
    }
}

fn the_readings_state_their_readouts_and_the_sinusoid() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
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
                model::T_FREQ_DIM,
                model::T_MAX_PERIOD,
                model::T_FLIP_SIN_COS,
                1.0
            )]
        );
    }
}

fn every_declared_lane_classifies_into_its_own_class_where_every_merge_resolves() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let classes = poem::resolve_classes(&plan).expect("every merge resolves");
        let _catalog = row(deployment);
        let mut seen: Vec<((&str, Option<&str>), usize)> = Vec::new();
        for (name, stream) in lanes(deployment) {
            let word = word(&plan, name, stream);
            let class = classes
                .class_of(word & classes.mask)
                .unwrap_or_else(|| panic!("{deployment}: a {name}/{stream:?} lane has no class"));
            seen.push(((name, stream), class));
        }
        let distinct: BTreeSet<usize> = seen.iter().map(|(_, class)| *class).collect();
        assert_eq!(
            distinct.len(),
            seen.len(),
            "{deployment}: two lanes share a class: {seen:?}"
        );
    }
}

fn the_refiners_attend_within_a_lane_and_the_trunk_within_the_group() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let d = dims(deployment);
        let selection_of = |id: ValueId| match &plan.values[id.0 as usize].def {
            Def::Input(RuntimeInput::Geometry { kind, .. }) => match kind {
                GeomKind::GroupIndptr { select } => ("group", *select),
                GeomKind::LaneIndptr { select } => ("lane", *select),
                other => panic!("{deployment}: a ragged CSR that is no indptr: {other:?}"),
            },
            other => panic!("{deployment}: a ragged CSR that is not a geometry input: {other:?}"),
        };
        let ragged: Vec<(&str, Selection, u32)> = plan
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Attn(Attn::Ragged {
                    q_indptr,
                    kv_indptr,
                    head_dim,
                    ..
                }) => {
                    assert_eq!(
                        q_indptr, kv_indptr,
                        "{deployment}: self-attention pairs one CSR"
                    );
                    let (kind, select) = selection_of(*q_indptr);
                    Some((kind, select, *head_dim))
                }
                _ => None,
            })
            .collect();
        let refiners = 2 * d.refiner_layers as usize;
        assert_eq!(
            ragged.len(),
            refiners + d.joint_layers as usize,
            "{deployment}"
        );
        assert!(ragged.iter().all(|(_, _, hd)| *hd == d.head_dim));
        let lane_reads = ragged.iter().filter(|(kind, _, _)| *kind == "lane").count();
        let group_reads = ragged
            .iter()
            .filter(|(kind, _, _)| *kind == "group")
            .count();
        assert_eq!(
            (lane_reads, group_reads),
            (refiners, d.joint_layers as usize),
            "{deployment}"
        );

        let (text, refine, denoise) = ("text", "refine", "denoise");
        let (_, joint, _) = ragged.iter().find(|(kind, _, _)| *kind == "group").unwrap();
        assert!(joint.holds(word(&plan, denoise, Some("image"))));
        assert!(joint.holds(word(&plan, denoise, Some("context"))));
        assert!(!joint.holds(word(&plan, refine, Some("context"))));
        if deployment == TURBO {
            assert!(!joint.holds(word(&plan, text, None)));
        }
        let mut lane_selects: Vec<Selection> = Vec::new();
        for (_, select, _) in ragged.iter().filter(|(kind, _, _)| *kind == "lane") {
            if !lane_selects.contains(select) {
                lane_selects.push(*select);
            }
        }
        assert_eq!(
            lane_selects.len(),
            2,
            "{deployment}: two refiner selections"
        );
        assert!(lane_selects.iter().any(|s| {
            s.holds(word(&plan, refine, Some("context")))
                && !s.holds(word(&plan, denoise, Some("image")))
                && !s.holds(word(&plan, denoise, Some("context")))
        }));
        assert!(lane_selects.iter().any(|s| {
            s.holds(word(&plan, denoise, Some("image")))
                && !s.holds(word(&plan, denoise, Some("context")))
                && !s.holds(word(&plan, refine, Some("context")))
        }));

        let prefills = plan
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
        let want = if deployment == TURBO {
            model::TE_LAYERS as usize
        } else {
            0
        };
        assert_eq!(prefills, want);
    }
}

fn the_dit_turns_three_interleaved_axes_and_the_encoder_the_whole_neox_head() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let d = dims(deployment);
        let axis_ropes: Vec<RopeRow> = plan
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Elemwise(Elemwise::RopeAxes {
                    dims,
                    thetas,
                    form,
                    rotary_dim,
                    head_dim,
                    ..
                }) => Some((*dims, *thetas, *form, *rotary_dim, *head_dim)),
                _ => None,
            })
            .collect();
        assert_eq!(
            axis_ropes.len(),
            2 * (2 * d.refiner_layers + d.joint_layers) as usize
        );
        for rope in &axis_ropes {
            assert_eq!(
                *rope,
                (
                    d.rope_dims,
                    [model::ROPE_THETA; 4],
                    RopeForm::Interleaved,
                    d.head_dim,
                    d.head_dim
                )
            );
        }
        let full: Vec<(u32, f32, bool)> = plan
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Elemwise(Elemwise::Rope {
                    head_dim,
                    theta,
                    k: Some(_),
                    turn: poem_ir::Turn::Full { interleaved, .. },
                    ..
                }) => Some((*head_dim, *theta, *interleaved)),
                _ => None,
            })
            .collect();
        let want = if deployment == TURBO {
            vec![(model::TE_HEAD_DIM, model::TE_THETA, false); model::TE_LAYERS as usize]
        } else {
            vec![]
        };
        assert_eq!(full, want, "{deployment}");
    }
}

fn the_modulation_is_a_per_lane_f32_scale_over_a_bf16_trunk() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let d = dims(deployment);
        let ty = |id: ValueId| plan.values[id.0 as usize].ty.clone();
        let (mut lane_scales, mut row_pads) = (0usize, 0usize);
        for node in &plan.nodes {
            let Operation::Elemwise(Elemwise::Modulate {
                x,
                m,
                lane_of_row,
                form,
                ..
            }) = &node.op
            else {
                continue;
            };
            assert_eq!(
                ty(*x),
                Ty::Tensor {
                    shape: vec![Dim::Tokens, Dim::Const(u64::from(d.dim))],
                    dtype: Dtype::Bf16,
                }
            );
            match lane_of_row {
                Some(lanes) => {
                    lane_scales += 1;
                    assert_eq!(*form, poem::ModulateForm::Scale);
                    assert_eq!(
                        ty(*m),
                        Ty::Tensor {
                            shape: vec![Dim::Lanes, Dim::Const(u64::from(d.dim))],
                            dtype: Dtype::F32,
                        }
                    );
                    assert_eq!(
                        plan.values[lanes.0 as usize].def,
                        Def::Input(RuntimeInput::Geometry {
                            space: 0,
                            kind: GeomKind::RequestOfToken
                        })
                    );
                }
                None => {
                    row_pads += 1;
                    assert_eq!(*form, poem::ModulateForm::ScaleShift);
                    assert_eq!(
                        ty(*m),
                        Ty::Tensor {
                            shape: vec![Dim::Tokens, Dim::Const(u64::from(2 * d.dim))],
                            dtype: Dtype::F32,
                        },
                        "{deployment}: the pad flag projects per row, in f32"
                    );
                }
            }
        }
        let modulated = (d.refiner_layers + d.joint_layers) as usize;
        assert_eq!(
            (lane_scales, row_pads),
            (2 * modulated + 1, 2),
            "{deployment}: the caption and the image each turn their pad rows into their pad token"
        );

        let mut folds = 0usize;
        for node in &plan.nodes {
            if !matches!(
                &node.op,
                Operation::Elemwise(Elemwise::ResidualAdd { gate: Some(_), .. })
            ) {
                continue;
            }
            let mut pairs = Vec::new();
            poem::Operands::aliases(&node.op, &mut pairs);
            assert_eq!(pairs.len(), 1, "a gated fold is in place on its residual");
            folds += 1;
        }
        assert_eq!(folds, 2 * modulated, "{deployment}");
    }
}

fn the_generative_facts_state_the_readings_the_schedule_and_the_latent_space() {
    for deployment in ROWS {
        let facts = row(deployment).generative().expect("facts");
        let names = trace(deployment, Platform::Cuda)
            .facts
            .declared("reading")
            .map(|s| s.values.clone())
            .unwrap_or_default();
        let want: Vec<&str> = if deployment == TURBO {
            vec!["text", "refine", "denoise", "vae.decode", "vae.encode"]
        } else {
            vec!["refine", "denoise"]
        };
        assert_eq!(names, want, "{deployment}");
        let latent = facts.latent.expect("a denoiser states its latent space");
        assert_eq!(latent.channels, model::CHANNELS);
        assert_eq!(latent.spatial_compression, model::SPATIAL_COMPRESSION);
        let schedule = facts.schedule.as_ref().expect("and its schedule");
        assert_eq!(
            (schedule.kind, schedule.shift, schedule.train_steps),
            (ScheduleKind::Flow, 3.0, 1000)
        );
        let want = [
            1.0,
            0.954_545_4,
            0.9,
            0.833_333_3,
            0.75,
            0.642_857_1,
            0.5,
            0.3,
        ];
        assert_eq!(schedule.pinned_sigmas.len(), 8);
        for (got, want) in schedule.pinned_sigmas.iter().zip(want) {
            assert!(
                (got - want).abs() < 1e-6,
                "{deployment}: sigma {got} is not {want}"
            );
        }
    }
}
