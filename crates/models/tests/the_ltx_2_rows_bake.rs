use std::collections::{BTreeMap, BTreeSet};

use model::{DENOISE, Dims, REFINE_AUDIO, REFINE_VIDEO, VAE_DECODE};
use models::{PortKind, ReadoutKind, ScheduleKind};
use poem::{
    Attention, Def, Dim, Dtype, Elementwise, GeomKind, Operands, Operation, Platform, RaggedMask,
    Request, RopeForm, RuntimeInput, Selection, Stream, Trace, Ty, ValueId, seam,
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
    match deployment {
        FLAGSHIP => Dims::ltx_2_5(),
        MINI => Dims::mini(),
        other => panic!("no dims for `{other}`"),
    }
}

fn word(plan: &Trace, reading: &str, stream: Stream) -> u64 {
    let request = Request::new(4, false).on_stream(stream).in_reading(reading);
    plan.facts.word(&request)
}

#[test]
fn the_ltx_2_rows_bake_every_case() {
    every_row_traces_on_every_platform_holding_nothing_between_fires();
    the_ports_the_trace_reads_are_the_ports_the_facts_declare();
    each_lane_the_facts_list_classifies_into_its_own_class();
    the_attentions_pair_as_the_architecture_says();
    every_rope_is_one_ladder_across_the_row();
    every_row_bakes_on_every_platform();
    the_generative_facts_state_the_readings_the_latent_and_the_schedule();
    every_block_table_folds_into_a_copy_of_the_vector_the_stack_shares();
    the_modulation_is_a_per_lane_f32_vector_over_a_bf16_trunk();
}

fn every_row_traces_on_every_platform_holding_nothing_between_fires() {
    for deployment in ROWS {
        for platform in PLATFORMS {
            let plan = trace(deployment, platform);
            assert!(
                !plan.nodes.is_empty(),
                "{deployment} {platform:?}: an empty plan"
            );
            assert!(
                plan.caches.is_empty(),
                "{deployment} {platform:?}: a denoiser and two connectors hold nothing between fires"
            );
            let seams: BTreeMap<&str, usize> =
                plan.seams.iter().fold(BTreeMap::new(), |mut acc, s| {
                    *acc.entry(s.seam.as_str()).or_default() += 1;
                    acc
                });
            assert_eq!(
                seams.get(seam::VELOCITY.name),
                Some(&1),
                "{deployment}: one velocity, planted on the merge of the two modalities"
            );
            assert_eq!(
                seams.get(seam::HIDDEN.name),
                Some(&2),
                "{deployment}: one hidden per connector"
            );
            assert!(
                !seams.contains_key(seam::OUT.name),
                "{deployment}: a denoiser has no logits, and `out` was planted anyway"
            );
            assert_eq!(
                seams.get(seam::PIXELS.name),
                is_flagship(deployment).then_some(&1),
                "{deployment}: one `pixels` planting iff the row carries the VAE decoder"
            );
        }
    }
}

fn is_flagship(deployment: &str) -> bool {
    deployment == FLAGSHIP
}

fn traced_ports(plan: &Trace) -> BTreeSet<(String, u8, u32)> {
    let mut traced = BTreeSet::new();
    for decl in &plan.values {
        let (kind, port, width) = match &decl.def {
            Def::Input(RuntimeInput::Latents { port, width }) => ("Latents", *port, *width),
            Def::Input(RuntimeInput::Context { port, width }) => ("Context", *port, *width),
            Def::Input(RuntimeInput::LaneVector { port, width }) => ("LaneVector", *port, *width),
            Def::Input(RuntimeInput::AxisPositions { port, axes }) => {
                ("AxisPositions", *port, u32::from(*axes))
            }
            Def::Input(RuntimeInput::Voxels { port, channels }) => ("Voxels", *port, *channels),
            _ => continue,
        };
        traced.insert((kind.to_string(), port, width));
    }
    traced
}

fn the_ports_the_trace_reads_are_the_ports_the_facts_declare() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let facts = row(deployment).generative.as_ref().expect("facts");
        let mut declared: BTreeSet<(String, u8, u32)> = BTreeSet::new();
        for reading in &facts.readings {
            for (index, port) in reading.ports_indexed() {
                declared.insert((format!("{:?}", port.kind), index, port.width));
            }
        }
        assert_eq!(
            traced_ports(&plan),
            declared,
            "{deployment}: the facts and the trace bind one list of ports"
        );

        let d = dims(deployment);
        let denoise = facts
            .readings
            .iter()
            .find(|r| r.name == "denoise")
            .unwrap_or_else(|| panic!("{deployment} declares no `denoise`"));
        let at = |name: &str| {
            let (index, port) = denoise
                .port(name)
                .unwrap_or_else(|| panic!("{deployment}: `denoise` declares no port `{name}`"));
            (index, port.kind, port.width, port.streams.clone())
        };
        assert_eq!(
            at("latents"),
            (
                model::port::LATENTS,
                PortKind::Latents,
                d.channels,
                vec![Stream::Video, Stream::Audio]
            ),
            "{deployment}: one latent rectangle for the two modalities"
        );
        assert_eq!(
            at("context"),
            (
                model::port::CONTEXT,
                PortKind::Context,
                d.cross_dim,
                vec![Stream::Context]
            )
        );
        assert_eq!(
            at("audio_context"),
            (
                model::port::AUDIO_CONTEXT,
                PortKind::Context,
                d.audio_cross_dim,
                vec![Stream::Reference]
            )
        );
        assert_eq!(
            at("timestep"),
            (model::port::TIMESTEP, PortKind::LaneVector, 1, vec![])
        );
        assert_eq!(
            at("positions"),
            (
                model::port::POSITIONS,
                PortKind::AxisPositions,
                u32::from(model::ROPE_AXES),
                vec![Stream::Video]
            )
        );
        assert_eq!(
            at("audio_positions"),
            (
                model::port::TIME_POSITIONS,
                PortKind::AxisPositions,
                1,
                vec![Stream::Audio]
            )
        );
        assert_eq!(denoise.readout, ReadoutKind::Velocity);
        assert_eq!(denoise.readout_width, d.channels);
        for name in ["refine.video", "refine.audio"] {
            let refine = facts.readings.iter().find(|r| r.name == name).unwrap();
            let (index, port) = refine.port("text").expect("the packed trunk rows");
            assert_eq!(
                (index, port.kind, port.width),
                (model::port::TEXT, PortKind::Latents, d.text_in()),
                "{deployment} {name}: a token-less reading states its rows through a latents port"
            );
            assert_eq!(refine.readout, ReadoutKind::Hidden);
        }
        assert_eq!(
            facts
                .readings
                .iter()
                .find(|r| r.name == "refine.video")
                .unwrap()
                .readout_width,
            d.cross_dim
        );
        assert_eq!(
            facts
                .readings
                .iter()
                .find(|r| r.name == "refine.audio")
                .unwrap()
                .readout_width,
            d.audio_cross_dim
        );
    }
}

fn each_lane_the_facts_list_classifies_into_its_own_class() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let classes = poem::resolve_classes(&plan)
            .unwrap_or_else(|why| panic!("{deployment}: a merge does not resolve: {why:?}"));
        let facts = row(deployment).generative.as_ref().expect("facts");
        let _catalog = row(deployment);
        let mut seen: Vec<((&str, Stream), usize)> = Vec::new();
        for reading in &facts.readings {
            for &stream in &reading.streams {
                let request = Request::new(4, false)
                    .on_stream(stream)
                    .in_reading(reading.name);
                let w = plan.facts.word(&request);
                let class = classes.class_of(w & classes.mask).unwrap_or_else(|| {
                    panic!("{deployment}: `{}`/{stream:?} has no class", reading.name)
                });
                seen.push(((reading.name, stream), class));
            }
        }
        let distinct: BTreeSet<usize> = seen.iter().map(|(_, class)| *class).collect();
        assert_eq!(
            distinct.len(),
            seen.len(),
            "{deployment}: two lanes share a class: {seen:?}"
        );
        assert_eq!(
            seen.len(),
            6 + usize::from(is_flagship(deployment)),
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
        let video = word(&plan, "denoise", Stream::Video);
        let audio = word(&plan, "denoise", Stream::Audio);
        let ctx = word(&plan, "denoise", Stream::Context);
        let actx = word(&plan, "denoise", Stream::Reference);
        let refine_v = word(&plan, "refine.video", Stream::Text);
        let refine_a = word(&plan, "refine.audio", Stream::Text);

        let mut pairs: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        let mut connector_reads = 0usize;
        for node in &plan.nodes {
            let Operation::Attention(Attention::Ragged {
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
                assert!(q_sel.holds(refine_v) || q_sel.holds(refine_a));
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
            let Operation::Elementwise(Elementwise::RopeAxes {
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
        let no_other = plan.nodes.iter().any(|node| {
            matches!(
                &node.op,
                Operation::Elementwise(Elementwise::RopeFull { .. })
                    | Operation::Elementwise(Elementwise::RopePartial { .. })
                    | Operation::Elementwise(Elementwise::RopeMrope { .. })
            )
        });
        assert!(!no_other, "{deployment}: this text turns one kind of rope");
    }
}

fn budget() -> poem_compiler::Budget {
    poem_compiler::Budget {
        max_lanes: 64,
        max_tokens: 4096,
        buckets: vec![64, 256, 1024, 4096],
        max_adapters: 0,
    }
}

fn every_row_bakes_on_every_platform() {
    for platform in PLATFORMS {
        for deployment in ROWS {
            let plan = trace(deployment, platform);
            let budgets = poem_compiler::Budgets::of(budget())
                .with_voxels(poem_compiler::VoxelLadder::new(256, 2));
            let compiled = poem_compiler::compile_axes(
                &plan,
                &budgets,
                &poem_compiler::DeviceProfile::default(),
            )
            .unwrap_or_else(|why| panic!("{platform:?}: `{deployment}` does not bake: {why}"));
            let tiled: usize = compiled.regions.iter().map(|r| r.nodes.len()).sum();
            assert_eq!(
                tiled,
                plan.nodes.len(),
                "{platform:?} `{deployment}`: the regions tile the node list once"
            );
            assert_eq!(
                compiled.voxels.is_some(),
                is_flagship(deployment),
                "{platform:?} `{deployment}`: a voxel plan iff the row carries the VAE"
            );
        }
    }
    let refused = poem_compiler::compile(
        &trace(FLAGSHIP, Platform::Cuda),
        &budget(),
        &poem_compiler::DeviceProfile::default(),
    );
    assert!(
        matches!(refused, Err(poem_compiler::Error::Unsized { .. })),
        "the flagship bakes against no voxel ladder: {refused:?}"
    );
}

fn the_generative_facts_state_the_readings_the_latent_and_the_schedule() {
    for deployment in ROWS {
        let facts = row(deployment).generative.as_ref().expect("facts");
        let d = dims(deployment);
        for (at, reading) in facts.readings.iter().enumerate() {
            assert_eq!(usize::from(reading.index), at, "{deployment}: dense from 0");
            assert!(
                !reading.has_kv && !reading.takes_tokens,
                "{deployment}: every reading here binds float ports alone"
            );
            assert!(
                reading.positions.is_none(),
                "{deployment}: LTX turns physical coordinates, which no convention names"
            );
        }
        let names: Vec<&str> = facts.readings.iter().map(|r| r.name).collect();
        let mut want = vec!["denoise", "refine.video", "refine.audio"];
        if is_flagship(deployment) {
            want.push("vae.decode");
        }
        assert_eq!(
            names, want,
            "{deployment}: the decode reading iff the row carries the VAE"
        );
        assert_eq!(usize::from(DENOISE), 0);
        assert_eq!(usize::from(REFINE_VIDEO), 1);
        assert_eq!(usize::from(REFINE_AUDIO), 2);
        assert_eq!(usize::from(VAE_DECODE), 3);
        let latent = facts.latent.expect("a latent space");
        assert_eq!(
            (
                latent.channels,
                latent.patch_t,
                latent.patch_h,
                latent.patch_w
            ),
            (d.channels, 1, 1, 1),
            "{deployment}: a token is one latent cell"
        );
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
        assert!(facts.max_rows >= 4096);
        validate(facts);
    }
}

fn validate(facts: &models::Generative) {
    for reading in &facts.readings {
        assert!(reading.readout_width > 0);
        if !reading.takes_tokens {
            assert!(
                reading
                    .ports
                    .iter()
                    .any(|port| matches!(port.kind, PortKind::Latents | PortKind::Voxels)),
                "a token-less reading states its rows through a latents or voxels port"
            );
        }
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
            let Operation::Elementwise(Elementwise::AddBias { out, .. }) = &node.op else {
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
            let Operation::Elementwise(Elementwise::Modulate {
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
                Operation::Elementwise(Elementwise::GatedResidualAdd { .. }) => {
                    let mut pairs = Vec::new();
                    node.op.aliases(&mut pairs);
                    assert_eq!(
                        pairs.len(),
                        1,
                        "{deployment}: a gated fold is in place on its residual"
                    );
                }
                Operation::Elementwise(Elementwise::GateSigmoidMulHeads {
                    head_dim,
                    scale,
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

/// The widths and constants ltx_2's package declares.
#[allow(dead_code)]
mod model {
    pub const PATCH_T: u32 = 1;
    pub const PATCH_H: u32 = 1;
    pub const PATCH_W: u32 = 1;

    pub const VAE_SPATIAL_COMPRESSION: u32 = 32;
    pub const VAE_TEMPORAL_COMPRESSION: u32 = 8;
    pub const VAE_Z: u32 = 128;

    pub const VAE_RGB: u32 = 3;
    pub const VAE_PATCH: u32 = 4;
    pub const VAE_EPS: f32 = 1e-8;
    pub const VAE_DECODER_DIMS: [u32; 5] = [1024, 512, 512, 256, 128];
    pub const VAE_MID_RESNETS: u32 = 2;
    pub const VAE_UP_RESNETS: [u32; 4] = [2, 4, 6, 4];
    pub const VAE_UP_STRIDES: [[u32; 3]; 4] = [[2, 2, 2], [2, 2, 2], [2, 1, 1], [1, 2, 2]];

    pub const T_FREQ_DIM: u32 = 256;
    pub const T_MAX_PERIOD: f32 = 10_000.0;
    pub const T_FLIP_SIN_COS: bool = true;
    pub const T_SCALE: f32 = 1.0;

    pub const NORM_EPS: f32 = 1e-6;

    pub const ROPE_THETA: f32 = 10_000.0;
    pub const ROPE_MAX_POS: [f32; 3] = [20.0, 2048.0, 2048.0];
    pub const AUDIO_ROPE_MAX_POS: f32 = 20.0;
    pub const CROSS_ROPE_MAX_POS: f32 = 20.0;
    pub const ROPE_AXES: u8 = 3;
    pub const AUDIO_ROPE_AXES: u8 = 1;

    pub const VIDEO_SCALE: [f32; 3] = [8.0, 32.0, 32.0];
    pub const AUDIO_SCALE: f32 = 4.0;
    pub const CAUSAL_OFFSET: f32 = 1.0;
    pub const AUDIO_SAMPLING_RATE: f32 = 16_000.0;
    pub const AUDIO_HOP: f32 = 160.0;

    pub const GATE_SCALE: f32 = 2.0;

    pub const MOD_SLICES: u32 = 9;
    pub const AV_SS_SLICES: u32 = 4;
    pub const AV_GATE_SLICES: u32 = 1;
    pub const PROMPT_SLICES: u32 = 2;
    pub const HEAD_SLICES: u32 = 2;

    pub const AV_GATE_TIMESTEP_SCALE: f32 = 1.0;

    pub const TRAIN_STEPS: u32 = 1000;
    pub const DISTILLED_SIGMAS: [f32; 8] = [
        1.0, 0.993_75, 0.987_5, 0.981_25, 0.975, 0.909_375, 0.725, 0.421_875,
    ];
    pub const STAGE2_SIGMAS: [f32; 3] = [0.909_375, 0.725, 0.421_875];

    pub const TEXT_LAYERS: u32 = 49;
    pub const TEXT_LEN: u32 = 1024;
    pub const CONN_REGISTERS: u32 = 128;
    pub const CONN_ROPE_BASE: f32 = 4096.0;
    pub const CONN_FF_MULT: u32 = 4;

    pub mod port {
        pub const LATENTS: u8 = 0;
        pub const CONTEXT: u8 = 0;
        pub const AUDIO_CONTEXT: u8 = 1;
        pub const TEXT: u8 = 1;
        pub const TIMESTEP: u8 = 0;
        pub const POSITIONS: u8 = 0;
        pub const TIME_POSITIONS: u8 = 1;
        pub const VOXELS: u8 = 0;
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Dims {
        pub layers: u32,
        pub heads: u32,
        pub head_dim: u32,
        pub audio_heads: u32,
        pub audio_head_dim: u32,
        pub channels: u32,
        pub cross_dim: u32,
        pub audio_cross_dim: u32,
        pub ff_mult: u32,
        pub caption: u32,
        pub conn_layers: u32,
    }

    impl Dims {
        #[must_use]
        pub const fn ltx_2_5() -> Dims {
            Dims {
                layers: 48,
                heads: 32,
                head_dim: 128,
                audio_heads: 32,
                audio_head_dim: 64,
                channels: 128,
                cross_dim: 4096,
                audio_cross_dim: 2048,
                ff_mult: 4,
                caption: 3840,
                conn_layers: 8,
            }
        }

        #[must_use]
        pub const fn mini() -> Dims {
            Dims {
                layers: 2,
                heads: 2,
                head_dim: 128,
                audio_heads: 2,
                audio_head_dim: 64,
                channels: 128,
                cross_dim: 256,
                audio_cross_dim: 128,
                ff_mult: 4,
                caption: 16,
                conn_layers: 1,
            }
        }

        #[must_use]
        pub const fn dim(&self) -> u32 {
            self.heads * self.head_dim
        }

        #[must_use]
        pub const fn audio_dim(&self) -> u32 {
            self.audio_heads * self.audio_head_dim
        }

        #[must_use]
        pub const fn av_inner(&self) -> u32 {
            self.audio_heads * self.audio_head_dim
        }

        #[must_use]
        pub const fn text_in(&self) -> u32 {
            self.caption * TEXT_LAYERS
        }

        #[must_use]
        pub fn sm_scale(&self) -> f32 {
            (self.head_dim as f32).sqrt().recip()
        }

        #[must_use]
        pub fn audio_sm_scale(&self) -> f32 {
            (self.audio_head_dim as f32).sqrt().recip()
        }

        #[must_use]
        pub const fn rope_dims(&self) -> [u32; 4] {
            let f = self.dim() / (2 * ROPE_AXES as u32);
            [2 * f, 2 * f, 2 * f, 0]
        }

        #[must_use]
        pub const fn audio_rope_dims(&self) -> [u32; 4] {
            [self.audio_dim(), 0, 0, 0]
        }

        #[must_use]
        pub const fn av_rope_dims(&self) -> [u32; 4] {
            [self.av_inner(), 0, 0, 0]
        }
    }

    #[must_use]
    pub const fn rope_pad(dim: u32, axes: u8) -> u32 {
        let axes = axes as u32;
        dim / 2 - axes * (dim / (2 * axes))
    }

    pub const DENOISE: u8 = 0;
    pub const REFINE_VIDEO: u8 = 1;
    pub const REFINE_AUDIO: u8 = 2;
    pub const VAE_DECODE: u8 = 3;
}
