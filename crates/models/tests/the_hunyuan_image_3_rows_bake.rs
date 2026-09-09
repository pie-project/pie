use std::collections::{BTreeMap, BTreeSet};

use model_dsl::{
    Attention, Classify, Def, Elementwise, Linear, Operation, Platform, Request, RopeForm,
    RuntimeInput, Stream, Trace, seam,
};
use models::hunyuan_image_3::forward::{DENOISE, ENCODE, Facts, IMAGE_IN, IMAGE_OUT};
use models::hunyuan_image_3::model::{self, Dims};
use models::{PortKind, ReadoutKind, ScheduleKind};

type RopeRow = ([u32; 4], [f32; 4], RopeForm, u32, u32);

const TP1: &str = "hunyuanimage3-80b-a13b-bf16-u8g64-kv-bf16";
const TP4: &str = "hunyuanimage3-80b-a13b-bf16-u8g64-kv-bf16-tp4";
const TP4_U4: &str = "hunyuanimage3-80b-a13b-bf16-u4g64-kv-bf16-tp4";
const MINI: &str = "hunyuanimage3-mini-bf16-kv-bf16";
const ROWS: [&str; 4] = [TP1, TP4, TP4_U4, MINI];

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
    if sku == MINI {
        Dims::mini()
    } else {
        Dims::flagship()
    }
}

fn ranks(sku: &str) -> u32 {
    row(sku).recipe.tp
}

#[test]
fn the_hunyuan_image_3_rows_bake_every_case() {
    every_row_traces_on_every_platform_with_the_caches_and_seams_it_states();
    the_ports_the_trace_reads_are_the_ports_the_facts_declare();
    each_lane_the_facts_list_classifies_into_its_own_class();
    every_rope_turns_the_whole_head_as_two_equal_blocks_in_the_split_form();
    the_canvas_reads_a_bidirectional_masked_attention_over_the_frozen_prefix();
    the_mixture_is_a_renormalised_top_k_over_the_whole_bank_beside_a_shared_expert();
    both_fact_columns_state_what_the_trace_does();
    every_row_bakes_on_every_platform_at_its_own_rank();
}

fn every_row_traces_on_every_platform_with_the_caches_and_seams_it_states() {
    for sku in ROWS {
        for platform in PLATFORMS {
            let plan = trace(sku, platform);
            assert!(!plan.nodes.is_empty(), "{sku} {platform:?}: an empty plan");
            let d = dims(sku);
            assert_eq!(
                plan.caches.len(),
                d.layers as usize,
                "{sku} {platform:?}: one kv row a layer and no state slab"
            );
            let seams: BTreeMap<&str, usize> =
                plan.seams.iter().fold(BTreeMap::new(), |mut acc, s| {
                    *acc.entry(s.seam.as_str()).or_default() += 1;
                    acc
                });
            assert_eq!(
                seams.get(seam::OUT.name),
                Some(&1),
                "{sku}: the AR phases read logits"
            );
            assert_eq!(
                seams.get(seam::HIDDEN.name),
                Some(&1),
                "{sku}: the canvas reads its trunk rows back"
            );
            assert_eq!(
                seams.get(seam::PIXELS.name),
                Some(&2),
                "{sku}: one pixels seam per image-head arm"
            );
            assert!(
                !seams.contains_key(seam::VELOCITY.name),
                "{sku}: the velocity is the `image.out` arm's PIXELS plane on the voxel axis"
            );
        }
    }
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
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let facts = row(sku).generative.as_ref().expect("generative facts");
        let mut declared: BTreeSet<(String, u8, u32)> = BTreeSet::new();
        for reading in &facts.readings {
            for (index, port) in reading.ports_indexed() {
                declared.insert((format!("{:?}", port.kind), index, port.width));
            }
        }
        assert_eq!(
            traced_ports(&plan),
            declared,
            "{sku}: the facts and the trace bind one list of ports"
        );

        let d = dims(sku);
        let denoise = facts
            .readings
            .iter()
            .find(|r| r.name == "denoise")
            .unwrap_or_else(|| panic!("{sku} declares no `denoise`"));
        let at = |reading: &models::ReadingFact, name: &str| {
            let (index, port) = reading
                .port(name)
                .unwrap_or_else(|| panic!("{sku}: `{}` declares no `{name}`", reading.name));
            (index, port.kind, port.width)
        };
        assert_eq!(
            at(denoise, "latents"),
            (model::port::ROWS, PortKind::Latents, d.hidden)
        );
        assert_eq!(
            at(denoise, "special"),
            (model::port::SPECIAL, PortKind::Latents, 1)
        );
        assert_eq!(
            at(denoise, "timestep"),
            (model::port::TIMESTEP, PortKind::LaneVector, 1)
        );
        assert_eq!(
            at(denoise, "positions"),
            (
                model::port::POSITIONS,
                PortKind::AxisPositions,
                u32::from(model::ROPE_AXES)
            )
        );
        assert_eq!(denoise.readout, ReadoutKind::Hidden);
        assert_eq!(denoise.readout_width, d.hidden);

        let image_in = &facts.readings[usize::from(IMAGE_IN)];
        assert_eq!(
            at(image_in, "latent"),
            (
                model::port::LATENT_VOXELS,
                PortKind::Voxels,
                model::LATENT_CHANNELS + model::T_FREQ_DIM
            )
        );
        assert_eq!(image_in.ports.len(), 1);
        assert_eq!(image_in.readout_width, d.hidden);
        let image_out = &facts.readings[usize::from(IMAGE_OUT)];
        assert_eq!(
            at(image_out, "rows"),
            (
                model::port::ROW_VOXELS,
                PortKind::Voxels,
                d.hidden + model::T_FREQ_DIM
            )
        );
        assert_eq!(image_out.ports.len(), 1);
        assert_eq!(image_out.readout, ReadoutKind::Pixels);
        assert_eq!(image_out.readout_width, model::LATENT_CHANNELS);
    }
}

fn each_lane_the_facts_list_classifies_into_its_own_class() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let classes = model_dsl::resolve_classes(&plan)
            .unwrap_or_else(|why| panic!("{sku}: a merge does not resolve: {why:?}"));
        let facts = row(sku).generative.as_ref().expect("generative facts");
        let catalog = row(sku);
        let mut seen: Vec<(String, usize)> = Vec::new();
        let lanes: Vec<(String, u8, Stream, u32)> = facts
            .readings
            .iter()
            .map(|r| {
                (
                    r.name.to_string(),
                    r.index,
                    *r.streams.first().expect("every reading names a stream"),
                    8,
                )
            })
            .chain(std::iter::once((
                "encode/decode".to_string(),
                ENCODE,
                Stream::Text,
                1,
            )))
            .collect();
        for (name, reading, stream, rows) in lanes {
            let request = Request::new(rows, false)
                .on_stream(stream)
                .in_reading(reading);
            let w = (catalog.classify)(&request);
            assert_eq!(w, Facts::of(&request).word(), "{sku} {name}");
            let class = classes
                .class_of(w & classes.mask)
                .unwrap_or_else(|| panic!("{sku}: `{name}` has no class"));
            seen.push((name, class));
        }
        let distinct: BTreeSet<usize> = seen.iter().map(|(_, class)| *class).collect();
        assert_eq!(
            distinct.len(),
            seen.len(),
            "{sku}: two lanes share a class: {seen:?}"
        );
        assert_eq!(seen.len(), 5, "{sku}: four readings and the AR decode step");
    }
}

fn every_rope_turns_the_whole_head_as_two_equal_blocks_in_the_split_form() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let d = dims(sku);
        let ropes: Vec<RopeRow> = plan
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
                _ => continue_none(),
            })
            .collect();
        assert_eq!(ropes.len(), 2 * d.layers as usize, "{sku}: q and k a layer");
        let half = d.head_dim / 2;
        for rope in &ropes {
            assert_eq!(
                *rope,
                (
                    [half, half, 0, 0],
                    [model::ROPE_THETA; 4],
                    RopeForm::Split,
                    d.head_dim,
                    d.head_dim
                ),
                "{sku}"
            );
        }
        assert_eq!(d.rope_dims(), [half, half, 0, 0], "{sku}");
        let scale = model::rope_x_scale(d.head_dim);
        assert!(
            scale < 1.0 && scale > 0.5,
            "{sku}: the x scale is theta^(-2/d), got {scale}"
        );
        let neox = plan.nodes.iter().any(|node| {
            matches!(
                &node.op,
                Operation::Elementwise(Elementwise::RopeFull { .. })
                    | Operation::Elementwise(Elementwise::RopePartial { .. })
            )
        });
        assert!(!neox, "{sku}: this family turns through `rope_axes` alone");
    }
}

fn continue_none<T>() -> Option<T> {
    None
}

fn the_canvas_reads_a_bidirectional_masked_attention_over_the_frozen_prefix() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let d = dims(sku);
        let (mut masked, mut prefill, mut decode, mut appends) = (0usize, 0usize, 0usize, 0usize);
        for node in &plan.nodes {
            match &node.op {
                Operation::Attention(Attention::Masked {
                    causal,
                    head_dim,
                    sm_scale,
                    ..
                }) => {
                    masked += 1;
                    assert!(
                        !causal,
                        "{sku}: the canvas lifts the causal bound; its mask carries the shape"
                    );
                    assert_eq!(*head_dim, d.head_dim);
                    assert!((sm_scale - d.sm_scale()).abs() < 1e-7);
                }
                Operation::Attention(Attention::Prefill { .. }) => prefill += 1,
                Operation::Attention(Attention::Decode { .. }) => decode += 1,
                Operation::Attention(Attention::KvAppend { .. }) => appends += 1,
                _ => {}
            }
        }
        let layers = d.layers as usize;
        assert_eq!(masked, layers, "{sku}: one masked read a layer");
        assert_eq!(prefill, layers, "{sku}: one causal prefill a layer");
        assert_eq!(decode, layers, "{sku}: one AR decode a layer");
        assert_eq!(appends, layers, "{sku}: one kv append a layer, arm-blind");
    }
}

fn the_mixture_is_a_renormalised_top_k_over_the_whole_bank_beside_a_shared_expert() {
    for sku in ROWS {
        let plan = trace(sku, Platform::Cuda);
        let d = dims(sku);
        let mut routers = 0usize;
        let mut selects = 0usize;
        let mut quant = 0usize;
        for node in &plan.nodes {
            match &node.op {
                Operation::Linear(Linear::MoeTopkSoftmax { experts, top_k, .. }) => {
                    routers += 1;
                    assert_eq!((*experts, *top_k), (d.experts, d.top_k), "{sku}");
                }
                Operation::Linear(Linear::MoeMatmulSelect { .. }) => selects += 1,
                Operation::Linear(Linear::MoeMatmulSelectQuant { .. }) => {
                    selects += 1;
                    quant += 1;
                }
                _ => {}
            }
        }
        assert_eq!(routers, d.layers as usize, "{sku}: one router a layer");
        assert_eq!(selects, 2 * d.layers as usize, "{sku}: gate_up and down");
        let quantized = row(sku).recipe.weights.len() > 1;
        assert_eq!(
            quant > 0,
            quantized,
            "{sku}: the routed banks are quantized on the flagship rows alone"
        );
        let swiglus = plan
            .nodes
            .iter()
            .filter(|node| matches!(&node.op, Operation::Linear(Linear::MlpSwiglu { .. })))
            .count();
        assert_eq!(swiglus, 2 * d.layers as usize, "{sku}: routed and shared");
    }
}

fn both_fact_columns_state_what_the_trace_does() {
    for sku in ROWS {
        let catalog = row(sku);
        let d = dims(sku);
        let canvas = catalog.diffusion.expect("a forward-diffusion row");
        assert_eq!(canvas.hidden, d.hidden, "{sku}");
        assert!(
            canvas.canvas > 0 && canvas.canvas.is_multiple_of(16),
            "{sku}"
        );
        assert_eq!(
            canvas.self_cond_taps, 0,
            "{sku}: the cross-step state is the KV cache, not a soft embedding"
        );

        let facts = catalog.generative.as_ref().expect("generative facts");
        for (at, reading) in facts.readings.iter().enumerate() {
            assert_eq!(usize::from(reading.index), at, "{sku}: dense from 0");
            assert!(
                reading.positions.is_none(),
                "{sku}: the 2-D nesting is the family's"
            );
        }
        let names: Vec<&str> = facts.readings.iter().map(|r| r.name).collect();
        assert_eq!(names, vec!["encode", "denoise", "image.in", "image.out"]);
        let encode = &facts.readings[usize::from(ENCODE)];
        assert!(encode.has_kv && encode.takes_tokens, "{sku}");
        assert_eq!(encode.readout, ReadoutKind::Logits);
        assert_eq!(encode.readout_width, d.vocab);
        let denoise = &facts.readings[usize::from(DENOISE)];
        assert!(
            denoise.has_kv && denoise.takes_tokens,
            "{sku}: the canvas is a sequence AND a float lane (design D10)"
        );
        for voxel in [IMAGE_IN, IMAGE_OUT] {
            let arm = &facts.readings[usize::from(voxel)];
            assert!(!arm.has_kv && !arm.takes_tokens, "{sku}: {}", arm.name);
        }
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
            (
                model::LATENT_CHANNELS,
                1,
                1,
                1,
                model::SPATIAL_COMPRESSION,
                1
            )
        );
        let schedule = facts.schedule.as_ref().expect("a schedule");
        assert_eq!(schedule.kind, ScheduleKind::Flow);
        assert_eq!(schedule.shift, model::FLOW_SHIFT);
        assert_eq!(schedule.train_steps, model::TRAIN_STEPS);
        assert!(
            schedule.pinned_sigmas.is_empty(),
            "{sku}: nothing is pinned"
        );
        assert!(facts.max_rows > canvas.canvas, "{sku}");
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

fn budget() -> model_compiler::Budget {
    model_compiler::Budget {
        max_lanes: 8,
        max_tokens: 8192,
        buckets: vec![64, 1024, 8192],
        max_adapters: 0,
    }
}

fn every_row_bakes_on_every_platform_at_its_own_rank() {
    for platform in PLATFORMS {
        for sku in ROWS {
            let plan = trace(sku, platform);
            let budgets = model_compiler::Budgets::of(budget())
                .with_voxels(model_compiler::VoxelLadder::new(8192, 4));
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
            assert!(
                compiled.voxels.is_some(),
                "{platform:?} `{sku}`: the image head is a voxel plan"
            );
        }
    }
    for sku in [TP1, TP4] {
        let plan = trace(sku, Platform::Cuda);
        let d = dims(sku);
        let reduces = plan
            .nodes
            .iter()
            .filter(|node| matches!(&node.op, Operation::Collective(_)))
            .count();
        let want = if ranks(sku) > 1 {
            2 * d.layers as usize
        } else {
            0
        };
        assert_eq!(reduces, want, "{sku}: one all-reduce per cut projection");
    }
}
