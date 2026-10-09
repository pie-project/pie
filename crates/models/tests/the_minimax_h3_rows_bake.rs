use std::collections::{BTreeMap, BTreeSet};

pub mod minimax_h3_dims;

use models::{PortKind, ReadoutKind, ScheduleKind};
use poem::{
    Attention, Def, Dim, Dtype, Elementwise, Operation, Platform, RaggedMask, Request, RopeForm,
    RuntimeInput, Stream, Trace, Ty, seam,
};

use minimax_h3_dims::{self as model, Dims};

const FLAGSHIP: &str = "minimax-h3-fl2va-bf16-kv-bf16";
const MINI: &str = "minimax-h3-mini-bf16-kv-bf16";
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
    every_row_traces_on_every_platform_with_the_encoder_it_declares();
    the_seams_are_the_two_float_readouts_and_never_logits();
    the_ports_the_trace_reads_are_the_ports_the_facts_declare();
    every_lane_the_facts_list_lands_in_a_class_where_the_merges_resolve();
    one_joint_read_per_block_and_one_lane_read_per_refiner();
    the_trunk_turns_three_neox_axes_and_the_refiner_turns_nothing();
    the_modality_gather_is_three_weight_blocks_and_one_column_slice();
    every_row_bakes_on_every_platform();
    the_sharded_worlds_trace_and_bake();
    the_generative_facts_state_the_readings_the_latent_and_two_shifts();
    the_modulation_is_a_lane_vector_and_every_gated_fold_folds_its_residual();
    the_gather_merges_four_lane_shaped_arms_back_onto_the_reading();
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

fn kind_code(kind: PortKind) -> u8 {
    match kind {
        PortKind::Latents => 0,
        PortKind::LaneVector => 1,
        PortKind::Context => 2,
        PortKind::AxisPositions => 3,
        PortKind::Voxels => 4,
    }
}

fn ports_read(plan: &Trace) -> BTreeSet<(u8, u8, u32)> {
    let mut found = BTreeSet::new();
    for value in &plan.values {
        let Def::Input(input) = &value.def else {
            continue;
        };
        let width = |ty: &Ty| match ty {
            Ty::Tensor { shape, .. } => match shape.as_slice() {
                [_, Dim::Const(width)] => u32::try_from(*width).unwrap_or(0),
                _ => 1,
            },
            _ => 0,
        };
        let entry = match input {
            RuntimeInput::Latents { port, .. } => {
                Some((PortKind::Latents, *port, width(&value.ty)))
            }
            RuntimeInput::LaneVector { port, width } => Some((PortKind::LaneVector, *port, *width)),
            RuntimeInput::Context { port, width } => Some((PortKind::Context, *port, *width)),
            RuntimeInput::AxisPositions { port, axes } => {
                Some((PortKind::AxisPositions, *port, u32::from(*axes)))
            }
            RuntimeInput::Voxels { port, channels } => Some((PortKind::Voxels, *port, *channels)),
            _ => None,
        };
        if let Some((kind, port, width)) = entry {
            found.insert((kind_code(kind), port, width));
        }
    }
    found
}

fn the_ports_the_trace_reads_are_the_ports_the_facts_declare() {
    for deployment in ROWS {
        let facts = row(deployment)
            .generative
            .as_ref()
            .expect("generative facts");
        let declared: BTreeSet<(u8, u8, u32)> = facts
            .readings
            .iter()
            .flat_map(|reading| {
                reading
                    .ports_indexed()
                    .map(|(index, port)| (kind_code(port.kind), index, port.width))
            })
            .collect();
        assert_eq!(
            ports_read(&trace(deployment, Platform::Cuda)),
            declared,
            "`{deployment}`: the trace's runtime inputs and the readings' ports"
        );
        for reading in &facts.readings {
            for (positional, port) in reading.ports_indexed() {
                let (by_name, _) = reading.port(&port.name).expect("a declared port by name");
                assert_eq!(
                    by_name, positional,
                    "`{deployment}` reading `{}` port `{}`: the positional index and the stated one",
                    reading.name, port.name
                );
            }
        }
        let denoise = facts
            .readings
            .iter()
            .find(|reading| reading.name == "denoise")
            .expect("a denoise reading");
        for (name, kind, index) in [
            ("latents", PortKind::Latents, model::port::LATENTS),
            ("reference", PortKind::Latents, model::port::REFERENCE),
            ("audio", PortKind::Latents, model::port::AUDIO),
            ("context", PortKind::Latents, model::port::CONTEXT),
            ("timestep", PortKind::LaneVector, model::port::TIMESTEP),
            ("positions", PortKind::AxisPositions, model::port::POSITIONS),
        ] {
            let (at, fact) = denoise.port(name).unwrap_or_else(|| {
                panic!("`{deployment}`: the denoise reading declares no `{name}` port")
            });
            assert_eq!(
                (fact.kind, at),
                (kind, index),
                "`{deployment}`: port `{name}`"
            );
        }
    }
}

fn lanes(deployment: &str) -> Vec<(&'static str, u8, Stream)> {
    let facts = row(deployment)
        .generative
        .as_ref()
        .expect("generative facts");
    facts
        .readings
        .iter()
        .flat_map(|reading| {
            let streams = if reading.streams.is_empty() {
                vec![Stream::Text]
            } else {
                reading.streams.clone()
            };
            streams
                .into_iter()
                .map(move |stream| (reading.name.as_str(), reading.index, stream))
        })
        .collect()
}

fn every_lane_the_facts_list_lands_in_a_class_where_the_merges_resolve() {
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
            Operation::Attention(Attention::Ragged { mask, .. }) => Some(*mask),
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
            .filter(|node| matches!(&node.op, Operation::Attention(Attention::Prefill { .. })))
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
                let name = format!("dit.block.{i}.adaln.{which}");
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
        let pairs: BTreeSet<(u32, usize)> = [
            Stream::Text,
            Stream::Video,
            Stream::Audio,
            Stream::Reference,
        ]
        .into_iter()
        .map(|s| (model::timestep_slot(s), model::modality(s)))
        .collect();
        assert_eq!(
            pairs.len(),
            4,
            "`{deployment}`: the four lanes gather four different modulation vectors"
        );
        let facts = row(deployment).generative.as_ref().expect("facts");
        let denoise = facts
            .readings
            .iter()
            .find(|reading| reading.name == "denoise")
            .expect("a denoise reading");
        let (_, timestep) = denoise.port("timestep").expect("a timestep port");
        assert_eq!(
            timestep.width,
            model::TIMESTEP_SLOTS,
            "`{deployment}`: the timestep port carries every unique timestep of the step"
        );
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
            let compiled =
                poem_compiler::compile(&plan, &budget(), &poem_compiler::DeviceProfile::default())
                    .unwrap_or_else(|why| {
                        panic!("{platform:?}: `{deployment}` does not bake: {why}")
                    });
            let tiled: usize = compiled.regions.iter().map(|r| r.nodes.len()).sum();
            assert_eq!(
                tiled,
                plan.nodes.len(),
                "{platform:?} `{deployment}`: the regions tile the node list once"
            );
        }
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
        poem_compiler::compile(&plan, &budget(), &poem_compiler::DeviceProfile::default())
            .unwrap_or_else(|why| panic!("`{deployment}` does not bake: {why}"));
        for which in 0..model::MODALITIES {
            let name = format!("dit.block.0.adaln.{which}");
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
        let facts = row(deployment).generative.as_ref().expect("facts");
        for (at, reading) in facts.readings.iter().enumerate() {
            assert_eq!(
                usize::from(reading.index),
                at,
                "`{deployment}`: readings are dense from 0 in index order"
            );
            assert!(
                reading.positions.is_none(),
                "`{deployment}` reading `{}`: H3's shared audio-tick time axis is not a stated \
                 convention",
                reading.name
            );
        }
        let names: Vec<&str> = facts.readings.iter().map(|r| r.name.as_str()).collect();
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
                latent.patch_t,
                latent.patch_h,
                latent.patch_w,
                latent.spatial_compression,
                latent.temporal_compression
            ),
            (24, 1, 2, 2, 16, 4),
            "`{deployment}`: the f16 t4 c24 latent under a (1, 2, 2) patch"
        );
        let schedule = facts.schedule.as_ref().expect("a schedule");
        assert_eq!(schedule.kind, ScheduleKind::Flow);
        assert_eq!(schedule.shift, model::VIDEO_SHIFT);
        assert_eq!(
            schedule
                .stream_shifts
                .iter()
                .find(|(s, _)| *s == Stream::Video),
            Some(&(Stream::Video, model::VIDEO_SHIFT)),
            "`{deployment}`: the video grid's shift"
        );
        assert_eq!(
            schedule
                .stream_shifts
                .iter()
                .find(|(s, _)| *s == Stream::Audio),
            Some(&(Stream::Audio, model::AUDIO_SHIFT)),
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
        let denoise = facts
            .readings
            .iter()
            .find(|reading| reading.name == "denoise")
            .expect("a denoise reading");
        assert_eq!(denoise.readout, ReadoutKind::Velocity);
        assert_eq!(denoise.readout_width, model::VIDEO_FEATURES);
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
                Operation::Elementwise(Elementwise::Modulate { m, lane_of_row, .. }) => {
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
                Operation::Elementwise(Elementwise::GatedResidualAdd { r, r_out, .. }) => {
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
            token_merges, 1,
            "`{deployment}`: one packed row sequence, joined once"
        );
    }
}
