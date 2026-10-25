use std::collections::{BTreeMap, BTreeSet};

use poem::generative::{ReadoutKind, ScheduleKind};
use poem::{Attn, Elemwise, Linear, Operation, Platform, Request, RopeForm, Stream, Trace, seam};

type RopeRow = ([u32; 4], [f32; 4], RopeForm, u32, u32);

use crate::hunyuan_image_3_dims::{self as model, DENOISE, Dims, ENCODE, IMAGE_IN, IMAGE_OUT};
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
    if deployment == MINI {
        Dims::mini()
    } else {
        Dims::flagship()
    }
}

fn ranks(deployment: &str) -> u32 {
    row(deployment).deploy.tp
}

#[test]
fn the_hunyuan_image_3_rows_bake_every_case() {
    every_row_traces_on_every_platform_with_the_caches_and_seams_it_states();
    each_reading_reads_out_what_its_facts_state();
    each_lane_the_facts_list_classifies_into_its_own_class();
    every_rope_turns_the_whole_head_as_two_equal_blocks_in_the_split_form();
    the_canvas_reads_a_bidirectional_masked_attention_over_the_frozen_prefix();
    the_mixture_is_a_renormalised_top_k_over_the_whole_bank_beside_a_shared_expert();
    both_fact_columns_state_what_the_trace_does();
    a_sharded_row_reduces_once_per_cut_projection();
}

fn every_row_traces_on_every_platform_with_the_caches_and_seams_it_states() {
    for deployment in ROWS {
        for platform in PLATFORMS {
            let plan = trace(deployment, platform);
            assert!(
                !plan.nodes.is_empty(),
                "{deployment} {platform:?}: an empty plan"
            );
            let d = dims(deployment);
            assert_eq!(
                plan.caches.len(),
                d.layers as usize,
                "{deployment} {platform:?}: one kv row a layer and no state slab"
            );
            let seams: BTreeMap<&str, usize> =
                plan.seams.iter().fold(BTreeMap::new(), |mut acc, s| {
                    *acc.entry(s.seam.as_str()).or_default() += 1;
                    acc
                });
            assert_eq!(
                seams.get(seam::OUT.name),
                Some(&1),
                "{deployment}: the AR phases read logits"
            );
            assert_eq!(
                seams.get(seam::HIDDEN.name),
                Some(&1),
                "{deployment}: the canvas reads its trunk rows back"
            );
            assert_eq!(
                seams.get(seam::PIXELS.name),
                Some(&2),
                "{deployment}: one pixels seam per image-head arm"
            );
            assert!(
                !seams.contains_key(seam::VELOCITY.name),
                "{deployment}: the velocity is the `image.out` arm's PIXELS plane on the voxel axis"
            );
        }
    }
}

fn each_reading_reads_out_what_its_facts_state() {
    for deployment in ROWS {
        let facts = row(deployment).generative().expect("generative facts");
        let d = dims(deployment);
        let denoise = facts
            .readings
            .iter()
            .find(|r| r.name == "denoise")
            .unwrap_or_else(|| panic!("{deployment} declares no `denoise`"));
        assert_eq!(denoise.readout, ReadoutKind::Hidden);
        assert_eq!(denoise.readout_width, d.hidden);
        let image_in = &facts.readings[usize::from(IMAGE_IN)];
        assert_eq!(image_in.readout_width, d.hidden);
        let image_out = &facts.readings[usize::from(IMAGE_OUT)];
        assert_eq!(image_out.readout, ReadoutKind::Pixels);
        assert_eq!(image_out.readout_width, model::LATENT_CHANNELS);
    }
}

fn each_lane_the_facts_list_classifies_into_its_own_class() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let classes = poem::resolve_classes(&plan)
            .unwrap_or_else(|why| panic!("{deployment}: a merge does not resolve: {why:?}"));
        let facts = row(deployment).generative().expect("generative facts");
        let _catalog = row(deployment);
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
            // A denoise lane attends its canvas through the mask its inferlet
            // sends with it.
            let request = Request::new(rows, reading == DENOISE)
                .on_stream(stream)
                .in_reading(reading_name(reading));
            let w = plan.facts.word(&request);
            let class = classes
                .class_of(w & classes.mask)
                .unwrap_or_else(|| panic!("{deployment}: `{name}` has no class"));
            seen.push((name, class));
        }
        let distinct: BTreeSet<usize> = seen.iter().map(|(_, class)| *class).collect();
        assert_eq!(
            distinct.len(),
            seen.len(),
            "{deployment}: two lanes share a class: {seen:?}"
        );
        assert_eq!(
            seen.len(),
            5,
            "{deployment}: four readings and the AR decode step"
        );
    }
}

fn every_rope_turns_the_whole_head_as_two_equal_blocks_in_the_split_form() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let d = dims(deployment);
        let ropes: Vec<RopeRow> = plan
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
                _ => continue_none(),
            })
            .collect();
        assert_eq!(
            ropes.len(),
            2 * d.layers as usize,
            "{deployment}: q and k a layer"
        );
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
                "{deployment}"
            );
        }
        assert_eq!(d.rope_dims(), [half, half, 0, 0], "{deployment}");
        let scale = model::rope_x_scale(d.head_dim);
        assert!(
            scale < 1.0 && scale > 0.5,
            "{deployment}: the x scale is theta^(-2/d), got {scale}"
        );
        let neox = plan.nodes.iter().any(|node| {
            matches!(
                &node.op,
                Operation::Elemwise(Elemwise::Rope {
                    k: Some(_),
                    turn: poem_ir::Turn::Full { .. },
                    ..
                }) | Operation::Elemwise(Elemwise::Rope {
                    k: Some(_),
                    turn: poem_ir::Turn::Partial { .. },
                    ..
                })
            )
        });
        assert!(
            !neox,
            "{deployment}: this family turns through `rope_axes` alone"
        );
    }
}

fn continue_none<T>() -> Option<T> {
    None
}

fn the_canvas_reads_a_bidirectional_masked_attention_over_the_frozen_prefix() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let d = dims(deployment);
        let (mut masked, mut prefill, mut decode, mut appends) = (0usize, 0usize, 0usize, 0usize);
        for node in &plan.nodes {
            match &node.op {
                Operation::Attn(Attn::Prefill {
                    causal,
                    head_dim,
                    sm_scale,
                    mask: Some(_),
                    selection: None,
                    bias: None,
                    lse: None,
                    ..
                }) => {
                    masked += 1;
                    assert!(
                        !causal,
                        "{deployment}: the canvas lifts the causal bound; its mask carries the shape"
                    );
                    assert_eq!(*head_dim, d.head_dim);
                    assert!((sm_scale - d.sm_scale()).abs() < 1e-7);
                }
                Operation::Attn(Attn::Prefill {
                    mask: None,
                    selection: None,
                    bias: None,
                    lse: None,
                    ..
                }) => prefill += 1,
                Operation::Attn(Attn::Decode {
                    selection: None,
                    bias: None,
                    lse: None,
                    ..
                }) => decode += 1,
                Operation::Attn(Attn::KvAppend { .. }) => appends += 1,
                _ => {}
            }
        }
        let layers = d.layers as usize;
        assert_eq!(masked, layers, "{deployment}: one masked read a layer");
        assert_eq!(prefill, layers, "{deployment}: one causal prefill a layer");
        assert_eq!(decode, layers, "{deployment}: one AR decode a layer");
        assert_eq!(
            appends, layers,
            "{deployment}: one kv append a layer, arm-blind"
        );
    }
}

fn the_mixture_is_a_renormalised_top_k_over_the_whole_bank_beside_a_shared_expert() {
    for deployment in ROWS {
        let plan = trace(deployment, Platform::Cuda);
        let d = dims(deployment);
        let mut routers = 0usize;
        let mut selects = 0usize;
        let mut quant = 0usize;
        for node in &plan.nodes {
            match &node.op {
                Operation::Linear(Linear::MoeRoute {
                    experts,
                    top_k,
                    scale: None,
                    score: poem_ir::RouteScore::Softmax,
                    ..
                }) => {
                    routers += 1;
                    assert_eq!((*experts, *top_k), (d.experts, d.top_k), "{deployment}");
                }
                Operation::Linear(Linear::MoeSelect {
                    bias: None,
                    quantized: false,
                    quant: None,
                    ..
                }) => selects += 1,
                Operation::Linear(Linear::MoeSelect {
                    bias: None,
                    quantized: true,
                    quant: None,
                    ..
                }) => {
                    selects += 1;
                    quant += 1;
                }
                _ => {}
            }
        }
        assert_eq!(
            routers, d.layers as usize,
            "{deployment}: one router a layer"
        );
        assert_eq!(
            selects,
            2 * d.layers as usize,
            "{deployment}: gate_up and down"
        );
        let quantized = row(deployment).deploy.weights.len() > 1;
        assert_eq!(
            quant > 0,
            quantized,
            "{deployment}: the routed banks are quantized on the flagship rows alone"
        );
        let swiglus = plan
            .nodes
            .iter()
            .filter(|node| {
                matches!(
                    &node.op,
                    Operation::Linear(Linear::Glu {
                        packed: Some(_),
                        gate: None,
                        weights: None,
                        limit: None,
                        act: poem_ir::GluAct::Swiglu,
                        ..
                    })
                )
            })
            .count();
        assert_eq!(
            swiglus,
            2 * d.layers as usize,
            "{deployment}: routed and shared"
        );
    }
}

fn both_fact_columns_state_what_the_trace_does() {
    for deployment in ROWS {
        let catalog = row(deployment);
        let d = dims(deployment);
        let canvas = catalog.diffusion().expect("a forward-diffusion row");
        assert_eq!(canvas.hidden, d.hidden, "{deployment}");
        assert!(
            canvas.canvas > 0 && canvas.canvas.is_multiple_of(16),
            "{deployment}"
        );

        let facts = catalog.generative().expect("generative facts");
        for (at, reading) in facts.readings.iter().enumerate() {
            assert_eq!(usize::from(reading.index), at, "{deployment}: dense from 0");
            assert!(
                reading.positions.is_none(),
                "{deployment}: the 2-D nesting is the family's"
            );
        }
        let names: Vec<&str> = facts.readings.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["encode", "denoise", "image.in", "image.out"]);
        let encode = &facts.readings[usize::from(ENCODE)];
        assert!(encode.has_kv && encode.takes_tokens, "{deployment}");
        assert_eq!(encode.readout, ReadoutKind::Logits);
        assert_eq!(encode.readout_width, d.vocab);
        let denoise = &facts.readings[usize::from(DENOISE)];
        assert!(
            denoise.has_kv && denoise.takes_tokens,
            "{deployment}: the canvas is a sequence AND a float lane (design D10)"
        );
        for voxel in [IMAGE_IN, IMAGE_OUT] {
            let arm = &facts.readings[usize::from(voxel)];
            assert!(
                !arm.has_kv && !arm.takes_tokens,
                "{deployment}: {}",
                arm.name
            );
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
            "{deployment}: nothing is pinned"
        );
        validate(&facts);
    }
}

fn validate(facts: &poem::generative::Generative) {
    for reading in &facts.readings {
        assert!(reading.readout_width > 0);
    }
}

fn a_sharded_row_reduces_once_per_cut_projection() {
    for deployment in [TP1, TP4] {
        let plan = trace(deployment, Platform::Cuda);
        let d = dims(deployment);
        let reduces = plan
            .nodes
            .iter()
            .filter(|node| matches!(&node.op, Operation::Collective(_)))
            .count();
        let want = if ranks(deployment) > 1 {
            2 * d.layers as usize
        } else {
            0
        };
        assert_eq!(
            reduces, want,
            "{deployment}: one all-reduce per cut projection"
        );
    }
}

/// The name of the reading the forward's code `reading` stands for.
fn reading_name(reading: u8) -> &'static str {
    match reading {
        ENCODE => "encode",
        DENOISE => "denoise",
        IMAGE_IN => "image.in",
        IMAGE_OUT => "image.out",
        other => panic!("no reading has code {other}"),
    }
}
