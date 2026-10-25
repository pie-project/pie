use std::collections::BTreeSet;

use crate::flux_2_dims;

use flux_2_dims as forward;
use flux_2_dims as model;
use poem::{
    Attn, Def, Dim, Dtype, Elemwise, GeomKind, Operands, Operation, Platform, Request, RopeForm,
    RuntimeInput, Trace, Ty, ValueId, seam,
};
use poem_ir::{Conv, InputAxis};

const KLEIN: &str = "flux2-klein-4b-bf16-kv-bf16";
const MINI: &str = "flux2-mini-bf16-kv-bf16";

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

fn seams(plan: &Trace) -> BTreeSet<&str> {
    plan.seams.iter().map(|s| s.seam.as_str()).collect()
}

#[test]
fn the_flux_2_rows_bake_every_case() {
    both_rows_trace_on_every_platform_with_the_seams_and_caches_they_declare();
    the_readings_state_their_streams_readouts_and_sinusoids();
    each_stream_of_the_denoise_reading_classifies_into_its_own_class();
    every_ragged_read_is_self_paired_over_the_group_csr();
    every_rope_turns_four_axes_of_the_whole_head_interleaved();
    the_schedule_is_the_goldens_and_the_mu_fit_reproduces_it();
    the_modulation_is_a_per_lane_f32_vector_over_a_bf16_trunk();
    the_latent_is_patched_in_the_graph_and_the_positions_are_its_cells();
}

fn the_latent_is_patched_in_the_graph_and_the_positions_are_its_cells() {
    let patch = [1, model::PACK, model::PACK];
    for deployment in [KLEIN, MINI] {
        let plan = trace(deployment, Platform::Cuda);
        let convs: Vec<&Conv> = plan
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Operation::Conv(op) => Some(op),
                _ => None,
            })
            .collect();
        let count = |f: &dyn Fn(&Conv) -> bool| convs.iter().filter(|op| f(op)).count();
        assert_eq!(
            (
                count(&|op| matches!(op, Conv::Patchify { p, .. } if *p == patch)),
                count(&|op| matches!(op, Conv::Unpatchify { p, .. } if *p == patch)),
            ),
            (1, 1),
            "{deployment}: the latent in, the velocity out"
        );
        let cells: Vec<u32> = convs
            .iter()
            .filter_map(|op| match op {
                Conv::Cells { axis, .. } => Some(*axis),
                _ => None,
            })
            .collect();
        assert_eq!(
            cells,
            vec![3],
            "{deployment}: a text row's position runs down the index axis"
        );
        let ports: BTreeSet<String> = plan
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
            format!("latents:voxel:{}", model::VAE_CHANNELS),
            "reference:lane:1".to_string(),
            "positions:row:4".to_string(),
        ] {
            assert!(ports.contains(&want), "{deployment}: {want} in {ports:?}");
        }
        let velocity = plan
            .seams
            .iter()
            .find(|s| s.seam == seam::VELOCITY.name)
            .expect("a velocity seam")
            .values[0];
        assert_eq!(
            plan.values[velocity.0 as usize].ty,
            Ty::Tensor {
                shape: vec![Dim::Voxels, Dim::Const(u64::from(model::VAE_CHANNELS))],
                dtype: Dtype::Bf16,
            },
            "{deployment}: the velocity reads out in the latent's own voxels"
        );
        assert_eq!(
            plan.facts.declared("positions").map(|s| s.values.clone()),
            Some(vec!["computed".to_string(), "given".to_string()])
        );
    }

    let plan = trace(MINI, Platform::Cuda);
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

fn the_readings_state_their_streams_readouts_and_sinusoids() {
    for (deployment, guidance) in [(KLEIN, false), (MINI, true)] {
        let plan = trace(deployment, Platform::Cuda);
        assert_eq!(
            plan.facts.declared("stream").map(|s| s.values.clone()),
            Some(vec!["text".into(), "image".into(), "reference".into()]),
            "{deployment}"
        );
        let mut sinusoids: Vec<(u32, f32, bool, f32)> = plan
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
        assert_eq!(sinusoids, want, "{deployment}");
    }

    let plan = trace(KLEIN, Platform::Cuda);
    let latent = plan
        .inputs
        .iter()
        .position(|i| i.name == "latent")
        .expect("the decoder's latent input") as u32;
    let voxel_port = plan.values.iter().any(|decl| {
        matches!(
            decl.def,
            Def::Input(RuntimeInput::Guest {
                id,
                axis: InputAxis::Voxel,
                width: model::VAE_CHANNELS,
                ..
            }) if id == latent
        )
    });
    assert!(
        voxel_port,
        "the decoder arm reads the VAE-native latent as voxels"
    );
}

fn each_stream_of_the_denoise_reading_classifies_into_its_own_class() {
    for deployment in [KLEIN, MINI] {
        let plan = trace(deployment, Platform::Cuda);
        let classes = poem::resolve_classes(&plan).expect("every merge resolves");
        let _row = row(deployment);

        let mut seen = Vec::new();
        for stream in ["text", "image", "reference"] {
            let request = Request::new(1, false)
                .switched("stream", stream)
                .switched("reading", "denoise");
            let word = plan.facts.word(&request);
            let class = classes
                .class_of(word & classes.mask)
                .unwrap_or_else(|| panic!("{deployment}: a {stream:?} lane has no class"));
            seen.push((stream, class));
        }
        let distinct: BTreeSet<usize> = seen.iter().map(|(_, class)| *class).collect();
        assert_eq!(
            distinct.len(),
            3,
            "{deployment}: two streams share a class: {seen:?}"
        );
    }
    let readings = |deployment: &str| {
        trace(deployment, Platform::Cuda)
            .facts
            .declared("reading")
            .map(|s| s.values.clone())
    };
    assert_eq!(readings(MINI), Some(vec!["denoise".to_string()]));
    assert_eq!(
        readings(KLEIN),
        Some(
            ["text", "denoise", "vae.decode", "vae.encode"]
                .map(String::from)
                .to_vec()
        )
    );
}

fn every_ragged_read_is_self_paired_over_the_group_csr() {
    for (deployment, dims) in [
        (KLEIN, model::Dims::klein_4b()),
        (MINI, model::Dims::mini()),
    ] {
        let plan = trace(deployment, Platform::Cuda);
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
        assert_eq!(
            ragged.len(),
            (dims.double_blocks + dims.single_blocks) as usize,
            "{deployment}: one ragged read per block"
        );
        let csr = ragged[0].0;
        for (q, kv, head_dim) in &ragged {
            assert_eq!(*head_dim, model::HEAD_DIM, "{deployment}");
            assert_eq!(
                (*q, *kv),
                (csr, csr),
                "{deployment}: one CSR, both sides, every block"
            );
        }
        let Def::Input(RuntimeInput::Geometry {
            kind: GeomKind::GroupIndptr { select },
            ..
        }) = &plan.values[csr.0 as usize].def
        else {
            panic!("{deployment}: the joint CSR is not a group indptr");
        };
        let word = |stream: &str| {
            plan.facts.word(
                &Request::new(1, false)
                    .switched("stream", stream)
                    .switched("reading", "denoise"),
            )
        };
        for stream in ["text", "image", "reference"] {
            assert!(
                select.holds(word(stream)),
                "{deployment}: {stream:?} is in the joint group"
            );
        }
        let text_reading = plan
            .facts
            .word(&Request::new(1, false).switched("reading", "text"));
        if deployment == KLEIN {
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
                Operation::Elemwise(Elemwise::Rope {
                    head_dim: model::TE_HEAD_DIM,
                    k: Some(_),
                    turn: poem_ir::Turn::Full {
                        interleaved: false,
                        ..
                    },
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

fn the_schedule_is_the_goldens_and_the_mu_fit_reproduces_it() {
    let facts = row(KLEIN).generative().unwrap();
    let schedule = facts
        .schedule
        .as_ref()
        .expect("a denoiser states its schedule");
    assert_eq!(schedule.kind, poem::generative::ScheduleKind::Flow);
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
        (model::VAE_CHANNELS, model::PACK, model::VAE_COMPRESSION)
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
            Operation::Elemwise(Elemwise::ResidualAdd { gate: Some(_), .. })
        ) {
            continue;
        }
        let mut pairs = Vec::new();
        node.op.aliases(&mut pairs);
        assert_eq!(pairs.len(), 1, "a gated fold is in place on its residual");
    }
}
