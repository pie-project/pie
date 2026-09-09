use std::collections::{BTreeMap, BTreeSet};

use model_dsl::{
    Attention, Classify, Def, Dim, Dtype, Elementwise, GeomKind, Operation, Platform, Request,
    RopeForm, RuntimeInput, Selection, Stream, Trace, Ty, ValueId, seam,
};
use models::z_image::forward::Facts;
use models::z_image::model::{self, Dims};
use models::{PortKind, ReadoutKind, ScheduleKind};

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

fn row(sku: &str) -> &'static models::Sku {
    models::sku(sku).unwrap_or_else(|| {
        let names: Vec<&str> = models::skus().map(|row| row.name.as_str()).collect();
        panic!("this build ships no `{sku}`; rows are {names:#?}")
    })
}

fn trace(sku: &str, platform: Platform) -> Trace {
    (row(sku).trace)(platform)
}

fn dims(sku: &str) -> Dims {
    if sku == TURBO {
        Dims::turbo()
    } else {
        Dims::mini()
    }
}

fn codes(sku: &str) -> (u8, u8, u8) {
    if sku == TURBO { (0, 1, 2) } else { (0, 0, 1) }
}

fn lanes(sku: &str) -> Vec<(&'static str, u8, Stream)> {
    let facts = row(sku)
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
                .map(move |stream| (reading.name, reading.index, *stream))
        })
        .collect()
}

fn word(reading: u8, stream: Stream) -> u64 {
    let request = Request::new(4, false).on_stream(stream).in_reading(reading);
    Facts::of(&request).word()
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
    for sku in ROWS {
        for platform in PLATFORMS {
            let plan = trace(sku, platform);
            assert!(!plan.nodes.is_empty(), "{sku} {platform:?}: an empty plan");
            let want = if sku == TURBO {
                model::TE_LAYERS as usize
            } else {
                0
            };
            assert_eq!(
                plan.caches.len(),
                want,
                "{sku} {platform:?}: the encoder's kv rows, and nothing else, are held"
            );
            let seams: BTreeMap<&str, usize> =
                plan.seams.iter().fold(BTreeMap::new(), |mut acc, s| {
                    *acc.entry(s.seam.as_str()).or_default() += 1;
                    acc
                });
            assert_eq!(
                seams.get(seam::VELOCITY.name),
                Some(&1),
                "{sku}: one velocity"
            );
            assert!(
                !seams.contains_key(seam::OUT.name),
                "{sku}: a denoiser has no logits, and `out` was planted anyway"
            );
            let hidden = if sku == TURBO { 2 } else { 1 };
            assert_eq!(
                seams.get(seam::HIDDEN.name),
                Some(&hidden),
                "{sku}: one `hidden` per reading that reads hidden rows back"
            );
            if sku == TURBO {
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
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let facts = row(sku).generative.as_ref().expect("facts");
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
            "{sku}: the facts and the trace bind one list of ports"
        );

        let reading = |name: &str| {
            facts
                .readings
                .iter()
                .find(|r| r.name == name)
                .unwrap_or_else(|| panic!("{sku} declares no reading `{name}`"))
        };
        let at = |reading: &models::ReadingFact, name: &str| {
            let (index, port) = reading.port(name).unwrap_or_else(|| {
                panic!(
                    "{sku}: reading `{}` declares no port `{name}`",
                    reading.name
                )
            });
            (index, port.kind, port.width)
        };
        let d = dims(sku);
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
                    "{sku}: {kind} port {index} is read at two widths ({have} and {width}); \
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
        if sku == TURBO {
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
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let classes = model_dsl::resolve_classes(&plan).expect("every merge resolves");
        let catalog = row(sku);
        let mut seen: Vec<((&str, Stream), usize)> = Vec::new();
        for (name, index, stream) in lanes(sku) {
            let request = Request::new(4, false).on_stream(stream).in_reading(index);
            let word = (catalog.classify)(&request);
            assert_eq!(word, Facts::of(&request).word(), "{sku} {name} {stream:?}");
            let class = classes
                .class_of(word & classes.mask)
                .unwrap_or_else(|| panic!("{sku}: a {name}/{stream:?} lane has no class"));
            seen.push(((name, stream), class));
        }
        let distinct: BTreeSet<usize> = seen.iter().map(|(_, class)| *class).collect();
        assert_eq!(
            distinct.len(),
            seen.len(),
            "{sku}: two lanes share a class: {seen:?}"
        );
    }
}

fn the_refiners_attend_within_a_lane_and_the_trunk_within_the_group() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let d = dims(sku);
        let selection_of = |id: ValueId| match &plan.values[id.0 as usize].def {
            Def::Input(RuntimeInput::Geometry { kind, .. }) => match kind {
                GeomKind::GroupIndptr { select } => ("group", *select),
                GeomKind::LaneIndptr { select } => ("lane", *select),
                other => panic!("{sku}: a ragged CSR that is no indptr: {other:?}"),
            },
            other => panic!("{sku}: a ragged CSR that is not a geometry input: {other:?}"),
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
                    assert_eq!(q_indptr, kv_indptr, "{sku}: self-attention pairs one CSR");
                    let (kind, select) = selection_of(*q_indptr);
                    Some((kind, select, *head_dim))
                }
                _ => None,
            })
            .collect();
        let refiners = 2 * d.refiner_layers as usize;
        assert_eq!(ragged.len(), refiners + d.joint_layers as usize, "{sku}");
        assert!(ragged.iter().all(|(_, _, hd)| *hd == d.head_dim));
        let lane_reads = ragged.iter().filter(|(kind, _, _)| *kind == "lane").count();
        let group_reads = ragged
            .iter()
            .filter(|(kind, _, _)| *kind == "group")
            .count();
        assert_eq!(
            (lane_reads, group_reads),
            (refiners, d.joint_layers as usize),
            "{sku}"
        );

        let (text, refine, denoise) = codes(sku);
        let (_, joint, _) = ragged.iter().find(|(kind, _, _)| *kind == "group").unwrap();
        assert!(joint.holds(word(denoise, Stream::Image)));
        assert!(joint.holds(word(denoise, Stream::Context)));
        assert!(!joint.holds(word(refine, Stream::Context)));
        if sku == TURBO {
            assert!(!joint.holds(word(text, Stream::Text)));
        }
        let mut lane_selects: Vec<Selection> = Vec::new();
        for (_, select, _) in ragged.iter().filter(|(kind, _, _)| *kind == "lane") {
            if !lane_selects.contains(select) {
                lane_selects.push(*select);
            }
        }
        assert_eq!(lane_selects.len(), 2, "{sku}: two refiner selections");
        assert!(lane_selects.iter().any(|s| {
            s.holds(word(refine, Stream::Context))
                && !s.holds(word(denoise, Stream::Image))
                && !s.holds(word(denoise, Stream::Context))
        }));
        assert!(lane_selects.iter().any(|s| {
            s.holds(word(denoise, Stream::Image))
                && !s.holds(word(denoise, Stream::Context))
                && !s.holds(word(refine, Stream::Context))
        }));

        let prefills = plan
            .nodes
            .iter()
            .filter(|node| matches!(&node.op, Operation::Attention(Attention::Prefill { .. })))
            .count();
        let want = if sku == TURBO {
            model::TE_LAYERS as usize
        } else {
            0
        };
        assert_eq!(prefills, want);
    }
}

fn the_dit_turns_three_interleaved_axes_and_the_encoder_the_whole_neox_head() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let d = dims(sku);
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
        let want = if sku == TURBO {
            vec![(model::TE_HEAD_DIM, model::TE_THETA, false); model::TE_LAYERS as usize]
        } else {
            vec![]
        };
        assert_eq!(full, want, "{sku}");
    }
}

fn the_modulation_is_a_per_lane_f32_scale_over_a_bf16_trunk() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let d = dims(sku);
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
                    assert_eq!(*form, model_dsl::ModulateForm::Scale);
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
                    assert_eq!(*form, model_dsl::ModulateForm::ScaleShift);
                    assert_eq!(
                        ty(*m),
                        Ty::Tensor {
                            shape: vec![Dim::Tokens, Dim::Const(u64::from(2 * d.dim))],
                            dtype: Dtype::Bf16,
                        },
                        "{sku}: the pad flag projects per row in the trunk's dtype"
                    );
                }
            }
        }
        let modulated = (d.refiner_layers + d.joint_layers) as usize;
        assert_eq!((lane_scales, row_pads), (2 * modulated + 1, 2), "{sku}");

        let mut folds = 0usize;
        for node in &plan.nodes {
            if !matches!(
                &node.op,
                Operation::Elementwise(Elementwise::GatedResidualAdd { .. })
            ) {
                continue;
            }
            let mut pairs = Vec::new();
            model_dsl::Operands::aliases(&node.op, &mut pairs);
            assert_eq!(pairs.len(), 1, "a gated fold is in place on its residual");
            folds += 1;
        }
        assert_eq!(folds, 2 * modulated, "{sku}");
    }
}

fn every_row_bakes() {
    for sku in ROWS {
        for platform in [Platform::Cuda, Platform::Metal] {
            let plan = trace(sku, platform);
            let max_tokens = if sku == TURBO { 8192 } else { 4096 };
            let budget = model_compiler::Budget {
                max_lanes: 64,
                max_tokens,
                buckets: (0..=13)
                    .map(|i| 1 << i)
                    .filter(|b| *b <= max_tokens)
                    .collect(),
                max_adapters: 0,
            };
            let budgets = model_compiler::Budgets::of(budget)
                .with_voxels(model_compiler::VoxelLadder::new(4096, 4));
            let compiled = model_compiler::compile_axes(
                &plan,
                &budgets,
                &model_compiler::DeviceProfile::default(),
            )
            .unwrap_or_else(|why| panic!("{sku} {platform:?}: does not bake: {why}"));
            assert!(
                !compiled.regions.is_empty(),
                "{sku} {platform:?}: a bake with no regions"
            );
            let tiled: usize = compiled
                .regions
                .iter()
                .map(|region| region.nodes.len())
                .sum();
            assert_eq!(
                tiled,
                plan.nodes.len(),
                "{sku} {platform:?}: the regions tile the nodes once"
            );
        }
    }
}

fn the_generative_facts_state_the_readings_the_schedule_and_the_latent_space() {
    for sku in ROWS {
        let facts = row(sku).generative.as_ref().expect("facts");
        let names: Vec<&str> = facts.readings.iter().map(|r| r.name).collect();
        let want: Vec<&str> = if sku == TURBO {
            vec!["text", "refine", "denoise", "vae.decode", "vae.encode"]
        } else {
            vec!["refine", "denoise"]
        };
        assert_eq!(names, want, "{sku}");
        for (at, reading) in facts.readings.iter().enumerate() {
            assert_eq!(
                usize::from(reading.index),
                at,
                "{sku}: indices dense from 0"
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
                "{sku}: sigma {got} is not {want}"
            );
        }
        assert!(facts.max_rows >= 4096);
    }
}
