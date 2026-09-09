use std::collections::BTreeSet;

use model_dsl::{
    Attention, Classify, Def, Dim, Dtype, Elementwise, GeomKind, Operands, Operation, Platform,
    Request, RopeForm, RuntimeInput, Stream, Trace, Ty, ValueId, seam,
};
use models::flux_2::forward::{self, Facts};
use models::flux_2::model;
use models::{PortKind, ReadoutKind};

const KLEIN: &str = "flux2-klein-4b-bf16-kv-bf16";
const MINI: &str = "flux2-mini-bf16-kv-bf16";

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

fn seams(plan: &Trace) -> BTreeSet<&str> {
    plan.seams.iter().map(|s| s.seam.as_str()).collect()
}

#[test]
fn the_flux_2_rows_bake_every_case() {
    both_rows_trace_on_every_platform_with_the_seams_and_caches_they_declare();
    the_denoise_ports_are_the_ones_the_facts_declare();
    each_stream_of_the_denoise_reading_classifies_into_its_own_class();
    every_ragged_read_is_self_paired_over_the_group_csr();
    every_rope_turns_four_axes_of_the_whole_head_interleaved();
    both_rows_bake_on_every_platform_under_a_voxel_ladder();
    the_schedule_is_the_goldens_and_the_mu_fit_reproduces_it();
    the_modulation_is_a_per_lane_f32_vector_over_a_bf16_trunk();
}

fn both_rows_trace_on_every_platform_with_the_seams_and_caches_they_declare() {
    for platform in PLATFORMS {
        let klein = trace(KLEIN, platform);
        assert_eq!(
            klein.caches.len(),
            model::TE_LAYERS as usize,
            "{platform:?}: the encoder's kv rows, one per layer it runs"
        );
        let klein_seams = seams(&klein);
        for want in [seam::HIDDEN.name, seam::VELOCITY.name, seam::PIXELS.name] {
            assert!(
                klein_seams.contains(want),
                "{platform:?}: the flagship reads out `{want}`; seams are {klein_seams:?}"
            );
        }
        assert!(
            !klein_seams.contains(seam::OUT.name),
            "{platform:?}: nothing here has logits, and `out` was planted anyway"
        );

        let mini = trace(MINI, platform);
        assert!(
            mini.caches.is_empty(),
            "{platform:?}: the miniature holds nothing between fires"
        );
        let mini_seams = seams(&mini);
        assert!(
            mini_seams.contains(seam::VELOCITY.name),
            "{platform:?}: {mini_seams:?}"
        );
        for absent in [seam::OUT.name, seam::HIDDEN.name, seam::PIXELS.name] {
            assert!(
                !mini_seams.contains(absent),
                "{platform:?}: a denoiser alone, and `{absent}` was planted"
            );
        }
    }
}

fn traced_ports(plan: &Trace) -> Vec<(PortKind, u8, u32)> {
    let mut traced: Vec<(PortKind, u8, u32)> = plan
        .values
        .iter()
        .filter_map(|decl| match &decl.def {
            Def::Input(RuntimeInput::Latents { port, width }) => {
                Some((PortKind::Latents, *port, *width))
            }
            Def::Input(RuntimeInput::Context { port, width }) => {
                Some((PortKind::Context, *port, *width))
            }
            Def::Input(RuntimeInput::LaneVector { port, width }) => {
                Some((PortKind::LaneVector, *port, *width))
            }
            Def::Input(RuntimeInput::AxisPositions { port, axes }) => {
                Some((PortKind::AxisPositions, *port, u32::from(*axes)))
            }
            _ => None,
        })
        .collect();
    traced.sort_by_key(|(kind, port, _)| (format!("{kind:?}"), *port));
    traced.dedup();
    traced
}

fn the_denoise_ports_are_the_ones_the_facts_declare() {
    for (sku, guidance, context_width) in [
        (KLEIN, false, model::Dims::klein_4b().dim),
        (MINI, true, model::Dims::mini().context_in),
    ] {
        let facts = row(sku)
            .generative
            .as_ref()
            .expect("a generative row states its readings");
        let denoise = facts
            .readings
            .iter()
            .find(|r| r.name == "denoise")
            .expect("the denoise reading");
        assert!(!denoise.has_kv && !denoise.takes_tokens, "{sku}");
        assert_eq!(
            denoise.streams,
            vec![Stream::Text, Stream::Image, Stream::Reference],
            "{sku}"
        );
        assert_eq!(denoise.readout, ReadoutKind::Velocity);
        assert_eq!(denoise.readout_width, model::IN_CHANNELS);
        let names: Vec<&str> = denoise.ports.iter().map(|p| p.name).collect();
        let want: Vec<&str> = if guidance {
            vec!["latents", "context", "timestep", "guidance", "positions"]
        } else {
            vec!["latents", "context", "timestep", "positions"]
        };
        assert_eq!(names, want, "{sku}");
        let at = |name: &str| {
            let (index, port) = denoise.port(name).unwrap();
            (index, port.kind, port.width)
        };
        assert_eq!(
            at("latents"),
            (model::port::LATENTS, PortKind::Latents, model::IN_CHANNELS)
        );
        assert_eq!(
            at("context"),
            (model::port::CONTEXT, PortKind::Context, context_width)
        );
        assert_eq!(
            at("timestep"),
            (model::port::TIMESTEP, PortKind::LaneVector, 1)
        );
        if guidance {
            assert_eq!(
                at("guidance"),
                (model::port::GUIDANCE, PortKind::LaneVector, 1)
            );
        }
        assert_eq!(
            at("positions"),
            (
                model::port::POSITIONS,
                PortKind::AxisPositions,
                u32::from(model::ROPE_AXES)
            )
        );

        let plan = trace(sku, Platform::Cuda);
        let mut declared: Vec<(PortKind, u8, u32)> = denoise
            .ports_indexed()
            .map(|(index, port)| (port.kind, index, port.width))
            .collect();
        declared.sort_by_key(|(kind, port, _)| (format!("{kind:?}"), *port));
        assert_eq!(traced_ports(&plan), declared, "{sku}");

        let mut sinusoids: Vec<(u32, f32, bool, f32)> = plan
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
        sinusoids.sort_by(|a, b| a.3.total_cmp(&b.3));
        let mut want = vec![(
            model::T_FREQ_DIM,
            model::T_MAX_PERIOD,
            model::T_FLIP_SIN_COS,
            model::T_SCALE,
        )];
        if guidance {
            want.push((
                model::T_FREQ_DIM,
                model::T_MAX_PERIOD,
                model::T_FLIP_SIN_COS,
                model::GUIDANCE_SCALE,
            ));
        }
        assert_eq!(sinusoids, want, "{sku}");
    }

    let klein = row(KLEIN).generative.as_ref().unwrap();
    let names: Vec<(&str, u8)> = klein.readings.iter().map(|r| (r.name, r.index)).collect();
    assert_eq!(
        names,
        vec![
            ("text", 0),
            ("denoise", 1),
            ("vae.decode", 2),
            ("vae.encode", 3)
        ]
    );
    let text = &klein.readings[0];
    assert!(text.has_kv && text.takes_tokens && text.ports.is_empty());
    assert_eq!(text.readout, ReadoutKind::Hidden);
    assert_eq!(text.readout_width, model::Dims::klein_4b().dim);
    let voxel_port = trace(KLEIN, Platform::Cuda).values.iter().any(|decl| {
        matches!(
            decl.def,
            Def::Input(RuntimeInput::Voxels {
                port: model::port::VOXELS,
                channels: model::IN_CHANNELS
            })
        )
    });
    assert!(
        voxel_port,
        "the decoder arm reads the packed latent as voxels"
    );
}

fn each_stream_of_the_denoise_reading_classifies_into_its_own_class() {
    for sku in [KLEIN, MINI] {
        let plan = trace(sku, Platform::Cuda);
        let classes = model_dsl::resolve_classes(&plan).expect("every merge resolves");
        let row = row(sku);
        let codes = models::flux_2::model::Model::mini(Dtype::Bf16, 1).readings();
        let denoise = if sku == KLEIN { 1 } else { codes.denoise };

        let mut seen = Vec::new();
        for stream in [Stream::Text, Stream::Image, Stream::Reference] {
            let request = Request::new(1, false).on_stream(stream).in_reading(denoise);
            let word = (row.classify)(&request);
            assert_eq!(word, Facts::of(&request).word(), "{sku} {stream:?}");
            let class = classes
                .class_of(word & classes.mask)
                .unwrap_or_else(|| panic!("{sku}: a {stream:?} lane has no class"));
            seen.push((stream, class));
        }
        let distinct: BTreeSet<usize> = seen.iter().map(|(_, class)| *class).collect();
        assert_eq!(
            distinct.len(),
            3,
            "{sku}: two streams share a class: {seen:?}"
        );
    }
    let mini = models::flux_2::model::Model::mini(Dtype::Bf16, 1).readings();
    assert_eq!(
        (mini.text, mini.denoise, mini.vae_decode, mini.vae_encode),
        (None, 0, None, None)
    );
    let klein = models::flux_2::model::Model::klein_4b(Dtype::Bf16, 1).readings();
    assert_eq!(
        (
            klein.text,
            klein.denoise,
            klein.vae_decode,
            klein.vae_encode
        ),
        (Some(0), 1, Some(2), Some(3))
    );
}

fn every_ragged_read_is_self_paired_over_the_group_csr() {
    for (sku, dims) in [
        (KLEIN, model::Dims::klein_4b()),
        (MINI, model::Dims::mini()),
    ] {
        let plan = trace(sku, Platform::Cuda);
        let ragged: Vec<(ValueId, ValueId, u32)> = plan
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Attention(Attention::Ragged {
                    q_indptr,
                    kv_indptr,
                    head_dim,
                    ..
                }) => Some((*q_indptr, *kv_indptr, *head_dim)),
                _ => None,
            })
            .collect();
        assert_eq!(
            ragged.len(),
            (dims.double_blocks + dims.single_blocks) as usize,
            "{sku}: one ragged read per block"
        );
        let csr = ragged[0].0;
        for (q, kv, head_dim) in &ragged {
            assert_eq!(*head_dim, model::HEAD_DIM, "{sku}");
            assert_eq!(
                (*q, *kv),
                (csr, csr),
                "{sku}: one CSR, both sides, every block"
            );
        }
        let Def::Input(RuntimeInput::Geometry {
            kind: GeomKind::GroupIndptr { select },
            ..
        }) = &plan.values[csr.0 as usize].def
        else {
            panic!("{sku}: the joint CSR is not a group indptr");
        };
        let denoise = if sku == KLEIN { 1 } else { 0 };
        let word = |stream: Stream| {
            Facts::of(&Request::new(1, false).on_stream(stream).in_reading(denoise)).word()
        };
        for stream in [Stream::Text, Stream::Image, Stream::Reference] {
            assert!(
                select.holds(word(stream)),
                "{sku}: {stream:?} is in the joint group"
            );
        }
        let text_reading = Facts::of(&Request::new(1, false).in_reading(0)).word();
        if sku == KLEIN {
            assert!(
                !select.holds(text_reading),
                "an encoder lane is not in the joint group"
            );
        }
    }
}

fn every_rope_turns_four_axes_of_the_whole_head_interleaved() {
    let plan = trace(KLEIN, Platform::Cuda);
    let dims = model::Dims::klein_4b();
    type Rope = ([u32; 4], [f32; 4], RopeForm, u32, u32);
    let ropes: Vec<Rope> = plan
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
        ropes.len(),
        (4 * dims.double_blocks + 2 * dims.single_blocks) as usize
    );
    for rope in &ropes {
        assert_eq!(
            *rope,
            (
                model::ROPE_DIMS,
                [model::ROPE_THETA; 4],
                RopeForm::Interleaved,
                model::HEAD_DIM,
                model::HEAD_DIM
            )
        );
    }
    let neox = plan
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                &node.op,
                Operation::Elementwise(Elementwise::RopeFull {
                    interleaved: false,
                    head_dim: model::TE_HEAD_DIM,
                    ..
                })
            )
        })
        .count();
    assert_eq!(
        neox,
        model::TE_LAYERS as usize,
        "the encoder's rope, once a layer"
    );
}

fn budget() -> model_compiler::Budget {
    model_compiler::Budget {
        max_lanes: 64,
        max_tokens: 4096,
        buckets: vec![64, 256, 1024, 4096],
        max_adapters: 0,
    }
}

fn both_rows_bake_on_every_platform_under_a_voxel_ladder() {
    for platform in PLATFORMS {
        for sku in [KLEIN, MINI] {
            let plan = trace(sku, platform);
            let budgets = model_compiler::Budgets::of(budget())
                .with_voxels(model_compiler::VoxelLadder::new(4096, 4));
            let compiled = model_compiler::compile_axes(
                &plan,
                &budgets,
                &model_compiler::DeviceProfile::default(),
            )
            .unwrap_or_else(|why| panic!("{platform:?}: `{sku}` does not bake: {why}"));
            let tiled: usize = compiled.regions.iter().map(|r| r.nodes.len()).sum();
            assert_eq!(
                tiled,
                plan.nodes.len(),
                "{platform:?} `{sku}`: the regions tile the node list once"
            );
            assert_eq!(
                compiled.voxels.is_some(),
                sku == KLEIN,
                "{platform:?} `{sku}`: a voxel plan iff the row carries a VAE"
            );
        }
    }
    let refused = model_compiler::compile(
        &trace(KLEIN, Platform::Cuda),
        &budget(),
        &model_compiler::DeviceProfile::default(),
    );
    assert!(
        matches!(refused, Err(model_compiler::Error::Unsized { .. })),
        "the flagship bakes against no voxel ladder: {refused:?}"
    );
}

fn the_schedule_is_the_goldens_and_the_mu_fit_reproduces_it() {
    let facts = row(KLEIN).generative.as_ref().unwrap();
    let schedule = facts
        .schedule
        .as_ref()
        .expect("a denoiser states its schedule");
    assert_eq!(schedule.kind, models::ScheduleKind::Flow);
    assert_eq!(schedule.train_steps, model::TRAIN_STEPS);
    let golden = [1.0f32, 0.967_384_04, 0.908_143_94, 0.767_199_93];
    assert_eq!(schedule.pinned_sigmas.len(), 4);
    for (mine, theirs) in schedule.pinned_sigmas.iter().zip(golden) {
        assert!((mine - theirs).abs() < 2e-6, "{mine} vs {theirs}");
    }
    assert_eq!(forward::sigmas(4096, 4), schedule.pinned_sigmas);
    assert_eq!(
        forward::empirical_mu(8192, 4),
        forward::empirical_mu(8192, 50)
    );
    assert_eq!(forward::sigmas(4096, 1), vec![1.0]);

    let latent = facts.latent.unwrap();
    assert_eq!(
        (latent.channels, latent.patch_h, latent.spatial_compression),
        (model::IN_CHANNELS, 1, model::TOKEN_COMPRESSION)
    );
    assert_eq!(
        model::VAE_CHANNELS * model::PACK * model::PACK,
        model::IN_CHANNELS,
        "a token is a 2×2 block of VAE latent cells"
    );
}

fn the_modulation_is_a_per_lane_f32_vector_over_a_bf16_trunk() {
    let plan = trace(MINI, Platform::Cuda);
    let dims = model::Dims::mini();
    let ty = |id: ValueId| plan.values[id.0 as usize].ty.clone();
    let mut seen = 0usize;
    for node in &plan.nodes {
        let Operation::Elementwise(Elementwise::Modulate {
            x, m, lane_of_row, ..
        }) = &node.op
        else {
            continue;
        };
        seen += 1;
        assert_eq!(
            ty(*x),
            Ty::Tensor {
                shape: vec![Dim::Tokens, Dim::Const(u64::from(dims.dim))],
                dtype: Dtype::Bf16,
            }
        );
        assert_eq!(
            ty(*m),
            Ty::Tensor {
                shape: vec![Dim::Lanes, Dim::Const(u64::from(2 * dims.dim))],
                dtype: Dtype::F32,
            }
        );
        assert!(lane_of_row.is_some(), "every modulation here is per lane");
    }
    assert_eq!(
        seen,
        (4 * dims.double_blocks + dims.single_blocks + 1) as usize
    );
    for node in &plan.nodes {
        if !matches!(
            &node.op,
            Operation::Elementwise(Elementwise::GatedResidualAdd { .. })
        ) {
            continue;
        }
        let mut pairs = Vec::new();
        node.op.aliases(&mut pairs);
        assert_eq!(pairs.len(), 1, "a gated fold is in place on its residual");
    }
}
