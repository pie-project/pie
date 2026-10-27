use std::collections::{BTreeMap, BTreeSet};

use crate::ltx_2_dims::{self as model, Dims};
use poem::generative::ScheduleKind;
use poem::{
    Attn, Def, Dim, Dtype, Elemwise, GeomKind, Operands, Operation, Platform, RaggedMask, Request,
    RopeForm, RuntimeInput, Selection, Trace, Ty, ValueId,
};

const FLAGSHIP: &str = "ltx25-bf16-kv-bf16";
const MINI: &str = "ltx25-mini-bf16-kv-bf16";
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
        FLAGSHIP => Dims::ltx_2_5(),
        MINI => Dims::mini(),
        other => panic!("no dims for `{other}`"),
    }
}

fn word(plan: &Trace, reading: &str, stream: Option<&str>) -> u64 {
    let mut request = Request::new(4, false).switched("reading", reading);
    if let Some(stream) = stream {
        request = request.switched("stream", stream);
    }
    plan.facts.word(&request)
}

#[test]
fn the_ltx_2_rows_bake_every_case() {
    every_row_traces_on_every_platform_holding_nothing_between_fires();
    each_lane_the_facts_list_classifies_into_its_own_class();
    the_attentions_pair_as_the_architecture_says();
    every_rope_is_one_ladder_across_the_row();
    the_generative_facts_state_the_readings_the_latent_and_the_schedule();
    every_block_table_folds_into_a_copy_of_the_vector_the_stack_shares();
    the_modulation_is_a_per_lane_f32_vector_over_a_bf16_trunk();
    the_latents_ride_the_voxels_and_the_positions_are_computed();
}

fn every_row_traces_on_every_platform_holding_nothing_between_fires() {
    for deployment in ROWS {
        for platform in PLATFORMS {
            let plan = trace(deployment, platform);
            assert!(
                !plan.nodes.is_empty(),
                "{deployment} {platform:?}: an empty plan"
            );
            assert_eq!(
                plan.caches.len(),
                if is_flagship(deployment) {
                    model::TEXT_LAYERS
                } else {
                    0
                },
                "{deployment} {platform:?}: the text encoder's kv, and nothing else between fires"
            );
            let seams: BTreeMap<&str, usize> =
                plan.outputs.iter().fold(BTreeMap::new(), |mut acc, s| {
                    *acc.entry(s.name.as_str()).or_default() += 1;
                    acc
                });
            assert_eq!(
                seams.get("velocity"),
                Some(&1),
                "{deployment}: one velocity, planted on the merge of the two modalities"
            );
            assert_eq!(
                seams.get("hidden"),
                is_flagship(deployment).then_some(&1),
                "{deployment}: the text reading's rows iff the row carries the encoder"
            );
            assert!(
                !seams.contains_key("logits"),
                "{deployment}: a denoiser has no logits, and `out` was planted anyway"
            );
            assert_eq!(
                seams.get("pixels"),
                is_flagship(deployment).then_some(&1),
                "{deployment}: one `pixels` planting iff the row carries the VAE decoder"
            );
        }
    }
}

fn is_flagship(deployment: &str) -> bool {
    deployment == FLAGSHIP
}

fn each_lane_the_facts_list_classifies_into_its_own_class() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let classes = poem::resolve_classes(&plan)
            .unwrap_or_else(|why| panic!("{deployment}: a merge does not resolve: {why:?}"));
        let mut lanes: Vec<(&str, Option<&str>)> = ["latents", "audio", "context", "audio_context"]
            .map(|stream| ("denoise", Some(stream)))
            .to_vec();
        if is_flagship(deployment) {
            lanes.extend([("text", None), ("vae.decode", None)]);
        }
        let mut seen: Vec<((&str, Option<&str>), usize)> = Vec::new();
        for (reading, stream) in lanes {
            let w = word(&plan, reading, stream);
            let class = classes
                .class_of(w & classes.mask)
                .unwrap_or_else(|| panic!("{deployment}: `{reading}`/{stream:?} has no class"));
            seen.push(((reading, stream), class));
        }
        let distinct: BTreeSet<usize> = seen.iter().map(|(_, class)| *class).collect();
        assert_eq!(
            distinct.len(),
            seen.len(),
            "{deployment}: two lanes share a class: {seen:?}"
        );
        assert_eq!(
            seen.len(),
            4 + 2 * usize::from(is_flagship(deployment)),
            "{deployment}: the lanes the facts list"
        );
    }
}

fn the_attentions_pair_as_the_architecture_says() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let d = dims(deployment);
        let selection_of = |id: ValueId| -> (Selection, &'static str) {
            match &plan.values[id.0 as usize].def {
                Def::Input(RuntimeInput::Geometry {
                    kind: GeomKind::GroupIndptr { select },
                    ..
                }) => (*select, "group"),
                Def::Input(RuntimeInput::Geometry {
                    kind: GeomKind::LaneIndptr { select },
                    ..
                }) => (*select, "lane"),
                other => panic!("{deployment}: a ragged CSR that is not an indptr: {other:?}"),
            }
        };
        let video = word(&plan, "denoise", Some("latents"));
        let audio = word(&plan, "denoise", Some("audio"));
        let ctx = word(&plan, "denoise", Some("context"));
        let actx = word(&plan, "denoise", Some("audio_context"));

        let mut pairs: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        let mut connector_reads = 0usize;
        for node in &plan.nodes {
            let Operation::Attn(Attn::Ragged {
                q_indptr,
                kv_indptr,
                head_dim,
                sm_scale,
                mask,
                ..
            }) = &node.op
            else {
                continue;
            };
            assert_eq!(*mask, RaggedMask::GroupBlockDiagonal, "{deployment}");
            let (q_sel, q_kind) = selection_of(*q_indptr);
            let (kv_sel, _) = selection_of(*kv_indptr);
            if q_kind == "lane" {
                connector_reads += 1;
                assert_eq!(
                    q_indptr, kv_indptr,
                    "{deployment}: a connector reads itself"
                );
                assert!(q_sel.holds(ctx) != q_sel.holds(actx));
                assert!(!q_sel.holds(video) && !q_sel.holds(audio));
                continue;
            }
            let name = |sel: &Selection| -> &'static str {
                match (
                    sel.holds(video),
                    sel.holds(audio),
                    sel.holds(ctx),
                    sel.holds(actx),
                ) {
                    (true, false, false, false) => "video",
                    (false, true, false, false) => "audio",
                    (false, false, true, false) => "context",
                    (false, false, false, true) => "audio_context",
                    other => panic!("{deployment}: a selection over {other:?}"),
                }
            };
            let (q, kv) = (name(&q_sel), name(&kv_sel));
            let want_head = match q {
                "video" if kv == "video" || kv == "context" => d.head_dim,
                _ => d.audio_head_dim,
            };
            assert_eq!(*head_dim, want_head, "{deployment}: {q} -> {kv}");
            let want_scale = (want_head as f32).sqrt().recip();
            assert!(
                (sm_scale - want_scale).abs() < 1e-7,
                "{deployment}: {q} -> {kv}"
            );
            *pairs.entry((q, kv)).or_default() += 1;
        }
        let layers = d.layers as usize;
        let want: BTreeMap<(&str, &str), usize> = [
            (("video", "video"), layers),
            (("audio", "audio"), layers),
            (("video", "context"), layers),
            (("audio", "audio_context"), layers),
            (("video", "audio"), layers),
            (("audio", "video"), layers),
        ]
        .into_iter()
        .collect();
        assert_eq!(pairs, want, "{deployment}: six attentions a block");
        assert_eq!(
            connector_reads,
            2 * d.conn_layers as usize,
            "{deployment}: one read per connector layer, twice over"
        );
    }
}

fn every_rope_is_one_ladder_across_the_row() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let d = dims(deployment);
        let mut seen: BTreeMap<([u32; 4], u32), usize> = BTreeMap::new();
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
            assert_eq!(*form, RopeForm::SplitLadder, "{deployment}");
            assert_eq!(*thetas, [model::ROPE_THETA; 4], "{deployment}");
            assert_eq!(
                rotary_dim, head_dim,
                "{deployment}: the ladder pairs rotate-half within a whole head"
            );
            *seen.entry((*dims, *head_dim)).or_default() += 1;
        }
        let layers = d.layers as usize;
        let conn = d.conn_layers as usize;
        let mut want: BTreeMap<([u32; 4], u32), usize> = BTreeMap::new();
        *want.entry((d.rope_dims(), d.head_dim)).or_default() += 2 * layers;
        *want
            .entry((d.audio_rope_dims(), d.audio_head_dim))
            .or_default() += 2 * layers;
        *want
            .entry((d.av_rope_dims(), d.audio_head_dim))
            .or_default() += 4 * layers;
        *want
            .entry(([d.cross_dim, 0, 0, 0], d.head_dim))
            .or_default() += 2 * conn;
        *want
            .entry(([d.audio_cross_dim, 0, 0, 0], d.audio_head_dim))
            .or_default() += 2 * conn;
        assert_eq!(seen, want, "{deployment}");

        assert_eq!(
            model::rope_pad(d.dim(), model::ROPE_AXES),
            2,
            "{deployment}"
        );
        assert_eq!(
            model::rope_pad(d.audio_dim(), model::AUDIO_ROPE_AXES),
            0,
            "{deployment}"
        );
        let encoder = plan
            .nodes
            .iter()
            .filter(|node| matches!(&node.op, Operation::Elemwise(Elemwise::Rope { .. })))
            .count();
        assert_eq!(
            encoder,
            if is_flagship(deployment) {
                model::TEXT_LAYERS
            } else {
                0
            },
            "{deployment}: the text encoder alone turns another rope, once a layer"
        );
    }
}

fn the_generative_facts_state_the_readings_the_latent_and_the_schedule() {
    for deployment in ROWS {
        let facts = row(deployment).generative().expect("facts");
        let d = dims(deployment);
        let names = trace(deployment, Platform::Cuda)
            .facts
            .declared("reading")
            .map(|s| s.values.clone())
            .unwrap_or_default();
        let want = if is_flagship(deployment) {
            vec!["text", "denoise", "vae.decode"]
        } else {
            vec!["denoise"]
        };
        assert_eq!(
            names, want,
            "{deployment}: the decode reading iff the row carries the VAE"
        );
        let latent = facts.latent.expect("a latent space");
        assert_eq!(latent.channels, d.channels, "{deployment}");
        assert_eq!(
            (latent.spatial_compression, latent.temporal_compression),
            (32, 8)
        );
        let schedule = facts.schedule.as_ref().expect("a schedule");
        assert_eq!(schedule.kind, ScheduleKind::Flow);
        assert_eq!(schedule.train_steps, model::TRAIN_STEPS);
        assert_eq!(schedule.boundary, None, "one backbone");
        assert_eq!(
            schedule.pinned_sigmas,
            model::DISTILLED_SIGMAS.to_vec(),
            "{deployment}: the distilled row pins eight sigmas for BOTH modalities"
        );
    }
}

fn every_block_table_folds_into_a_copy_of_the_vector_the_stack_shares() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let lane_shaped = |id: ValueId| {
            matches!(
                &plan.values[id.0 as usize].ty,
                Ty::Tensor { shape, .. } if shape.first() == Some(&Dim::Lanes)
            )
        };
        let mut folded = 0usize;
        for node in &plan.nodes {
            let Operation::Elemwise(Elemwise::AddBias { out, .. }) = &node.op else {
                continue;
            };
            if !lane_shaped(*out) {
                continue;
            }
            let readers = plan
                .nodes
                .iter()
                .filter(|other| {
                    let mut ins = Vec::new();
                    other.op.inputs(&mut ins);
                    ins.contains(out)
                })
                .count();
            assert_eq!(
                readers,
                1,
                "{deployment}: a bias folded in place on a lane vector {} other nodes also read",
                readers - 1
            );
            folded += 1;
        }
        let d = dims(deployment);
        assert!(
            folded >= 8 * d.layers as usize,
            "{deployment}: {folded} lane-vector folds for {} blocks",
            d.layers
        );
    }
}

fn the_modulation_is_a_per_lane_f32_vector_over_a_bf16_trunk() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let d = dims(deployment);
        let ty = |id: ValueId| plan.values[id.0 as usize].ty.clone();
        let mut modulates = 0usize;
        for node in &plan.nodes {
            let Operation::Elemwise(Elemwise::Modulate {
                x, m, lane_of_row, ..
            }) = &node.op
            else {
                continue;
            };
            modulates += 1;
            let width = match ty(*x) {
                Ty::Tensor { shape, dtype } => {
                    assert_eq!(dtype, Dtype::Bf16, "{deployment}: a bf16 trunk");
                    assert_eq!(shape[0], Dim::Tokens, "{deployment}");
                    match shape[1] {
                        Dim::Const(w) => w,
                        other => panic!("{deployment}: a modulated row of width {other:?}"),
                    }
                }
                other => panic!("{deployment}: {other:?}"),
            };
            assert_eq!(
                ty(*m),
                Ty::Tensor {
                    shape: vec![Dim::Lanes, Dim::Const(2 * width)],
                    dtype: Dtype::F32,
                },
                "{deployment}: a per-lane f32 scale/shift pair as wide as its rows"
            );
            let lanes = lane_of_row.expect("every modulation here is per lane");
            assert_eq!(
                plan.values[lanes.0 as usize].def,
                Def::Input(RuntimeInput::Geometry {
                    space: 0,
                    kind: GeomKind::RequestOfToken
                }),
                "{deployment}: the broadcast is the fire's token->lane table"
            );
        }
        assert_eq!(modulates, 12 * d.layers as usize + 2, "{deployment}");

        let mut gates = 0usize;
        for node in &plan.nodes {
            match &node.op {
                Operation::Elemwise(Elemwise::ResidualAdd { gate: Some(_), .. }) => {
                    let mut pairs = Vec::new();
                    node.op.aliases(&mut pairs);
                    assert_eq!(
                        pairs.len(),
                        1,
                        "{deployment}: a gated fold is in place on its residual"
                    );
                }
                Operation::Elemwise(Elemwise::GateSigmoidMul {
                    heads: Some(poem_ir::GateHeads { head_dim, scale }),
                    ..
                }) => {
                    gates += 1;
                    assert_eq!(
                        *scale,
                        model::GATE_SCALE,
                        "{deployment}: `out * 2 sigmoid(W x)`"
                    );
                    assert!(
                        *head_dim == d.head_dim || *head_dim == d.audio_head_dim,
                        "{deployment}: a gate logit per head"
                    );
                }
                _ => {}
            }
        }
        assert_eq!(
            gates,
            6 * d.layers as usize + 2 * d.conn_layers as usize,
            "{deployment}: every attention ends in a per-head gate"
        );
    }
}

fn the_latents_ride_the_voxels_and_the_positions_are_computed() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let d = dims(deployment);
        let inputs: BTreeSet<String> = plan
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
        for want in [
            format!("latents:voxel:{}", d.channels),
            "fps:lane:1".to_string(),
            "text_positions:row:1".to_string(),
            format!("context:row:{}", d.cross_dim + d.audio_cross_dim),
            "length:lane:1".to_string(),
            format!("positions:row:{}", model::ROPE_AXES),
            format!("audio_positions:row:{}", model::AUDIO_ROPE_AXES),
        ] {
            assert!(
                inputs.contains(&want),
                "{deployment}: no `{want}` in {inputs:?}"
            );
        }
        let context = plan.inputs.iter().find(|i| i.name == "context");
        assert_eq!(
            context.and_then(|i| i.rows),
            Some(poem_ir::RowCount::Exact(model::TEXT_LEN)),
            "{deployment}: the prompt padded to the reference's rows"
        );
        assert!(
            !inputs.iter().any(|i| i.starts_with("latents:row")),
            "{deployment}: the latent rides the voxel axis only"
        );
        let convs: Vec<&poem_ir::Conv> = plan
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Conv(op) => Some(op),
                _ => None,
            })
            .collect();
        let patch = [1, 1, 1];
        let count = |f: &dyn Fn(&poem_ir::Conv) -> bool| convs.iter().filter(|op| f(op)).count();
        assert_eq!(
            count(&|op| matches!(op, poem_ir::Conv::Patchify { p, .. } if *p == patch)),
            2,
            "{deployment}: the video and the audio clip patched in"
        );
        assert_eq!(
            count(&|op| matches!(op, poem_ir::Conv::Unpatchify { p, .. } if *p == patch)),
            2,
            "{deployment}: and their velocities patched back out"
        );
        assert_eq!(
            count(&|op| matches!(op, poem_ir::Conv::Cells { axis: 0, .. })),
            4,
            "{deployment}: one cell table a medium, and one a connector's prompt rows"
        );
        assert_eq!(
            plan.facts.declared("positions").map(|s| s.values.clone()),
            Some(vec!["computed".to_string(), "given".to_string()])
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
                shape: vec![Dim::Voxels, Dim::Const(u64::from(d.channels))],
                dtype: Dtype::Bf16,
            },
            "{deployment}: the velocity reads out in the latents' own voxels"
        );
        let budgets = poem_compiler::Budgets::of(poem_compiler::Budget::new(8, 1024)).with_voxels(
            poem_compiler::VoxelLadder {
                max_voxels: 4096,
                buckets: vec![1024, 4096],
                max_clips: 8,
            },
        );
        poem_compiler::compile_axes(&plan, &budgets, &poem_compiler::DeviceProfile::default())
            .unwrap_or_else(|why| panic!("{deployment} does not bake: {why}"));
    }
}
