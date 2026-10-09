use std::collections::{BTreeMap, BTreeSet};

pub mod z_image_dims;

use models::{PortKind, ReadoutKind, ScheduleKind};
use poem::{
    Attention, Def, Dim, Dtype, Elementwise, GeomKind, Operation, Platform, Request, RopeForm,
    RuntimeInput, Selection, Stream, Trace, Ty, ValueId, seam,
};
use z_image_dims::{self as model, Dims};

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

fn row(deployment: &str) -> &'static models::Deployment {
    models::deployment(deployment).unwrap_or_else(|| {
        let names: Vec<&str> = models::deployments().map(|row| row.name.as_str()).collect();
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

fn lanes(deployment: &str) -> Vec<(&'static str, u8, Stream)> {
    let facts = row(deployment)
        .generative
        .as_ref()
        .expect("a generative row states its facts");
    facts
        .readings
        .iter()
        .flat_map(|reading| {
            reading
                .streams
                .iter()
                .map(move |stream| (reading.name.as_str(), reading.index, *stream))
        })
        .collect()
}

fn word(plan: &Trace, reading: &str, stream: Stream) -> u64 {
    let request = Request::new(4, false).on_stream(stream).in_reading(reading);
    plan.facts.word(&request)
}

#[test]
fn the_z_image_rows_bake_every_case() {
    every_row_traces_on_every_platform_with_the_caches_and_readouts_it_states();
    the_ports_the_trace_reads_are_the_ports_the_facts_declare();
    every_declared_lane_classifies_into_its_own_class_where_every_merge_resolves();
    the_refiners_attend_within_a_lane_and_the_trunk_within_the_group();
    the_dit_turns_three_interleaved_axes_and_the_encoder_the_whole_neox_head();
    the_modulation_is_a_per_lane_f32_scale_over_a_bf16_trunk();
    every_row_bakes();
    the_generative_facts_state_the_readings_the_schedule_and_the_latent_space();
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

fn the_ports_the_trace_reads_are_the_ports_the_facts_declare() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let facts = row(deployment).generative.as_ref().expect("facts");
        let mut traced: BTreeSet<(String, u8, u32)> = BTreeSet::new();
        for decl in &plan.values {
            let (kind, port, width) = match &decl.def {
                Def::Input(RuntimeInput::Latents { port, width }) => ("Latents", *port, *width),
                Def::Input(RuntimeInput::Context { port, width }) => ("Context", *port, *width),
                Def::Input(RuntimeInput::LaneVector { port, width }) => {
                    ("LaneVector", *port, *width)
                }
                Def::Input(RuntimeInput::AxisPositions { port, axes }) => {
                    ("AxisPositions", *port, u32::from(*axes))
                }
                Def::Input(RuntimeInput::Voxels { port, channels }) => ("Voxels", *port, *channels),
                _ => continue,
            };
            traced.insert((kind.to_string(), port, width));
        }
        let mut declared: BTreeSet<(String, u8, u32)> = BTreeSet::new();
        for reading in &facts.readings {
            for (index, port) in reading.ports_indexed() {
                declared.insert((format!("{:?}", port.kind), index, port.width));
            }
        }
        assert_eq!(
            traced, declared,
            "{deployment}: the facts and the trace bind one list of ports"
        );

        let reading = |name: &str| {
            facts
                .readings
                .iter()
                .find(|r| r.name == name)
                .unwrap_or_else(|| panic!("{deployment} declares no reading `{name}`"))
        };
        let at = |reading: &models::ReadingFact, name: &str| {
            let (index, port) = reading.port(name).unwrap_or_else(|| {
                panic!(
                    "{deployment}: reading `{}` declares no port `{name}`",
                    reading.name
                )
            });
            (index, port.kind, port.width)
        };
        let d = dims(deployment);
        let axes = u32::from(model::ROPE_AXES);
        let denoise = reading("denoise");
        assert_eq!(
            at(denoise, "latents"),
            (
                model::port::LATENTS,
                PortKind::Latents,
                model::PATCH_FEATURES
            )
        );
        assert_eq!(
            at(denoise, "pad"),
            (model::port::PAD_IMAGE, PortKind::Latents, 1)
        );
        assert_eq!(
            at(denoise, "context"),
            (model::port::CONTEXT_REFINED, PortKind::Latents, d.dim)
        );
        assert_eq!(
            at(denoise, "timestep"),
            (model::port::TIMESTEP, PortKind::LaneVector, 1)
        );
        assert_eq!(
            at(denoise, "positions"),
            (model::port::POSITIONS, PortKind::AxisPositions, axes)
        );
        let refine = reading("refine");
        assert_eq!(
            at(refine, "pad"),
            (model::port::PAD_CAPTION, PortKind::Latents, 1)
        );
        assert_eq!(
            at(refine, "caption"),
            (model::port::CAPTION, PortKind::Context, d.cap_width)
        );

        let mut widths: std::collections::BTreeMap<(String, u8), u32> =
            std::collections::BTreeMap::new();
        for (kind, index, width) in &traced {
            if let Some(have) = widths.insert((kind.clone(), *index), *width) {
                assert_eq!(
                    have, *width,
                    "{deployment}: {kind} port {index} is read at two widths ({have} and {width}); \
                     the engine seats one rectangle per (kind, index)"
                );
            }
        }
        assert_eq!(
            at(refine, "positions"),
            (model::port::POSITIONS, PortKind::AxisPositions, axes)
        );
        let streams = |reading: &models::ReadingFact, name: &str| {
            reading.port(name).unwrap().1.streams.clone()
        };
        assert_eq!(streams(denoise, "latents"), vec![Stream::Image]);
        assert_eq!(streams(denoise, "pad"), vec![Stream::Image]);
        assert_eq!(streams(denoise, "context"), vec![Stream::Context]);
        assert_eq!(
            streams(denoise, "timestep"),
            vec![Stream::Image, Stream::Context]
        );
        assert_eq!(
            streams(denoise, "positions"),
            vec![Stream::Image, Stream::Context]
        );
        assert!(!refine.has_kv && !refine.takes_tokens, "a float lane");
        assert!(!denoise.has_kv && !denoise.takes_tokens, "float lanes");
        assert_eq!(refine.readout, ReadoutKind::Hidden);
        assert_eq!(refine.readout_width, d.dim);
        assert_eq!(denoise.readout, ReadoutKind::Velocity);
        assert_eq!(denoise.readout_width, model::PATCH_FEATURES);
        if deployment == TURBO {
            let text = reading("text");
            assert!(text.has_kv && text.takes_tokens && text.ports.is_empty());
            assert_eq!(
                (text.readout, text.readout_width),
                (ReadoutKind::Hidden, model::TE_HIDDEN)
            );
        }

        let sinusoids: Vec<(u32, f32, bool, f32)> = plan
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Elementwise(Elementwise::Sinusoid {
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
        let mut seen: Vec<((&str, Stream), usize)> = Vec::new();
        for (name, _index, stream) in lanes(deployment) {
            let request = Request::new(4, false).on_stream(stream).in_reading(name);
            let word = plan.facts.word(&request);
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
                Operation::Attention(Attention::Ragged {
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
        assert!(joint.holds(word(&plan, denoise, Stream::Image)));
        assert!(joint.holds(word(&plan, denoise, Stream::Context)));
        assert!(!joint.holds(word(&plan, refine, Stream::Context)));
        if deployment == TURBO {
            assert!(!joint.holds(word(&plan, text, Stream::Text)));
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
            s.holds(word(&plan, refine, Stream::Context))
                && !s.holds(word(&plan, denoise, Stream::Image))
                && !s.holds(word(&plan, denoise, Stream::Context))
        }));
        assert!(lane_selects.iter().any(|s| {
            s.holds(word(&plan, denoise, Stream::Image))
                && !s.holds(word(&plan, denoise, Stream::Context))
                && !s.holds(word(&plan, refine, Stream::Context))
        }));

        let prefills = plan
            .nodes
            .iter()
            .filter(|node| matches!(&node.op, Operation::Attention(Attention::Prefill { .. })))
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
                Operation::Elementwise(Elementwise::RopeAxes {
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
                Operation::Elementwise(Elementwise::RopeFull {
                    head_dim,
                    theta,
                    interleaved,
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
            let Operation::Elementwise(Elementwise::Modulate {
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
                            dtype: Dtype::Bf16,
                        },
                        "{deployment}: the pad flag projects per row in the trunk's dtype"
                    );
                }
            }
        }
        let modulated = (d.refiner_layers + d.joint_layers) as usize;
        assert_eq!(
            (lane_scales, row_pads),
            (2 * modulated + 1, 2),
            "{deployment}"
        );

        let mut folds = 0usize;
        for node in &plan.nodes {
            if !matches!(
                &node.op,
                Operation::Elementwise(Elementwise::GatedResidualAdd { .. })
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

fn every_row_bakes() {
    for deployment in ROWS {
        for platform in [Platform::Cuda, Platform::Metal] {
            let plan = trace(deployment, platform);
            let max_tokens = if deployment == TURBO { 8192 } else { 4096 };
            let budget = poem_compiler::Budget {
                max_lanes: 64,
                max_tokens,
                buckets: (0..=13)
                    .map(|i| 1 << i)
                    .filter(|b| *b <= max_tokens)
                    .collect(),
                max_adapters: 0,
            };
            let budgets = poem_compiler::Budgets::of(budget)
                .with_voxels(poem_compiler::VoxelLadder::new(4096, 4));
            let compiled = poem_compiler::compile_axes(
                &plan,
                &budgets,
                &poem_compiler::DeviceProfile::default(),
            )
            .unwrap_or_else(|why| panic!("{deployment} {platform:?}: does not bake: {why}"));
            assert!(
                !compiled.regions.is_empty(),
                "{deployment} {platform:?}: a bake with no regions"
            );
            let tiled: usize = compiled
                .regions
                .iter()
                .map(|region| region.nodes.len())
                .sum();
            assert_eq!(
                tiled,
                plan.nodes.len(),
                "{deployment} {platform:?}: the regions tile the nodes once"
            );
        }
    }
}

fn the_generative_facts_state_the_readings_the_schedule_and_the_latent_space() {
    for deployment in ROWS {
        let facts = row(deployment).generative.as_ref().expect("facts");
        let names: Vec<&str> = facts.readings.iter().map(|r| r.name.as_str()).collect();
        let want: Vec<&str> = if deployment == TURBO {
            vec!["text", "refine", "denoise", "vae.decode", "vae.encode"]
        } else {
            vec!["refine", "denoise"]
        };
        assert_eq!(names, want, "{deployment}");
        for (at, reading) in facts.readings.iter().enumerate() {
            assert_eq!(
                usize::from(reading.index),
                at,
                "{deployment}: indices dense from 0"
            );
        }
        let latent = facts.latent.expect("a denoiser states its latent space");
        assert_eq!(
            (
                latent.channels,
                latent.patch_t,
                latent.patch_h,
                latent.patch_w
            ),
            (model::CHANNELS, 1, model::PATCH, model::PATCH)
        );
        assert_eq!(latent.spatial_compression, model::SPATIAL_COMPRESSION);
        assert_eq!(
            latent.channels * latent.patch_h * latent.patch_w,
            model::PATCH_FEATURES,
            "a latent row is exactly the head's output row"
        );
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
        assert!(facts.max_rows >= 4096);
    }
}
