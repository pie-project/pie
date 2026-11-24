//! flux2-mini as engine-cpu's flux test and the goldens' generator draw it: its weights and a step's
//! inputs, by name, at `READING_ROWS` rows a lane (default 64), and its weights as a diffusers
//! pipeline's checkpoint (`dit.`, `te.` and `vae.`).

#![allow(dead_code)]

pub mod guest;

use std::path::Path;

use engine_cpu::Forward;
use engine_cpu::forward::{StreamStep, Values};
use poem_ir::{Operands, Trace};

pub const MINI: &str = "flux2-mini-bf16-kv-bf16";
pub const CHANNELS: usize = 32;
pub const CONTEXT: usize = 192;

/// The text lane's rows (and the image lane's patches).
pub fn rows() -> usize {
    std::env::var("READING_ROWS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(64)
}

/// The latent's voxel grid.
pub fn grid() -> [u32; 3] {
    let side = 2 * (rows() as f64).sqrt().round() as u32;
    [1, side, side]
}

fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// `count` values in [-1/2, 1/2), each exact in f32: what the goldens' generator draws for `name`.
pub fn units(name: &str, count: usize) -> Vec<f32> {
    let mut seed = fnv(name);
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 40) as i64 - (1 << 23)) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

pub fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

pub fn trace() -> Trace {
    poem_compiler::catalog::repository()
        .parse(MINI)
        .expect("the catalog holds flux2-mini")
        .trace(poem_ir::Platform::Cerebras)
}

/// The param `name`'s values (`count` of them): a norm's gain near one, else uniform with the
/// variance of 1 / fan-in.
pub fn plane(trace: &Trace, name: &str, count: usize) -> Vec<f32> {
    let u = units(name, count);
    if name.ends_with("norm") {
        return u.iter().map(|u| 1.0 + 0.5 * u).collect();
    }
    let fan_in = (trace.params.iter())
        .find(|p| p.name == name)
        .and_then(|p| p.shape.last().copied())
        .unwrap_or(1) as f32;
    let s = (12.0 / fan_in).sqrt();
    u.iter().map(|u| u * s).collect()
}

/// A step's guest inputs by name, as the trace's input axes lay them (sigma 0.5, guidance 4).
pub fn inputs() -> Vec<(String, Vec<f32>)> {
    let voxels = (grid()[1] * grid()[2]) as usize;
    let mut context = context();
    context.resize((rows() + voxels / 4) * CONTEXT, 0.0);
    vec![
        ("timestep".to_string(), vec![500.0; 2]),
        ("guidance".to_string(), vec![4.0; 2]),
        ("context".to_string(), context),
        ("latents".to_string(), latents()),
    ]
}

/// The text lane's context rows, `rows() x CONTEXT`.
pub fn context() -> Vec<f32> {
    let s = 12f32.sqrt();
    units("input.context", rows() * CONTEXT)
        .iter()
        .map(|u| bf16(u * s))
        .collect()
}

/// The latent at the first sigma, `voxels x CHANNELS`.
pub fn latents() -> Vec<f32> {
    let voxels = (grid()[1] * grid()[2]) as usize;
    let s = 12f32.sqrt();
    units("input.latents", voxels * CHANNELS)
        .iter()
        .map(|u| bf16(u * s))
        .collect()
}

/// A plane's shape and values.
type Plane = (Vec<usize>, Vec<f32>);

/// A VAE plane by its checkpoint name, as the goldens' generator draws it.
pub fn vae_plane(name: &str, shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let u = units(name, n);
    if name.ends_with("running_var") || ((name.contains("norm")) && name.ends_with(".weight")) {
        return u.iter().map(|u| 1.0 + 0.5 * u).collect();
    }
    if name.ends_with("running_mean") {
        return u.iter().map(|u| 0.5 * u).collect();
    }
    if name.ends_with(".bias") {
        return u.iter().map(|u| 0.2 * u).collect();
    }
    let fan: usize = shape[1..].iter().product::<usize>().max(1);
    let s = (12.0 / fan as f32).sqrt();
    u.iter().map(|u| u * s).collect()
}

/// The VAE's planes as diffusers' `AutoencoderKLFlux2` names them, and whether each is held in
/// f32 (norms, biases, the batch norm) rather than bf16.
pub fn vae_names() -> Vec<(String, Vec<usize>, bool)> {
    let c = 32usize;
    let mut v: Vec<(String, Vec<usize>, bool)> = Vec::new();
    let conv =
        |v: &mut Vec<(String, Vec<usize>, bool)>, n: String, o: usize, i: usize, k: usize| {
            v.push((format!("{n}.weight"), vec![o, i, k, k], false));
            v.push((format!("{n}.bias"), vec![o], true));
        };
    let norm = |v: &mut Vec<(String, Vec<usize>, bool)>, n: String| {
        v.push((format!("{n}.weight"), vec![c], true));
        v.push((format!("{n}.bias"), vec![c], true));
    };
    let resnet = |v: &mut Vec<(String, Vec<usize>, bool)>, n: String| {
        norm(v, format!("{n}.norm1"));
        conv(v, format!("{n}.conv1"), c, c, 3);
        norm(v, format!("{n}.norm2"));
        conv(v, format!("{n}.conv2"), c, c, 3);
    };
    let mid = |v: &mut Vec<(String, Vec<usize>, bool)>, n: &str| {
        resnet(v, format!("{n}.mid_block.resnets.0"));
        resnet(v, format!("{n}.mid_block.resnets.1"));
        let a = format!("{n}.mid_block.attentions.0");
        norm(v, format!("{a}.group_norm"));
        for p in ["to_q", "to_k", "to_v", "to_out.0"] {
            v.push((format!("{a}.{p}.weight"), vec![c, c], false));
            v.push((format!("{a}.{p}.bias"), vec![c], true));
        }
    };
    conv(&mut v, "encoder.conv_in".into(), c, 3, 3);
    for i in 0..4 {
        resnet(&mut v, format!("encoder.down_blocks.{i}.resnets.0"));
        if i < 3 {
            conv(
                &mut v,
                format!("encoder.down_blocks.{i}.downsamplers.0.conv"),
                c,
                c,
                3,
            );
        }
    }
    mid(&mut v, "encoder");
    norm(&mut v, "encoder.conv_norm_out".into());
    conv(&mut v, "encoder.conv_out".into(), 2 * c, c, 3);
    conv(&mut v, "decoder.conv_in".into(), c, c, 3);
    mid(&mut v, "decoder");
    for i in 0..4 {
        for r in 0..2 {
            resnet(&mut v, format!("decoder.up_blocks.{i}.resnets.{r}"));
        }
        if i < 3 {
            conv(
                &mut v,
                format!("decoder.up_blocks.{i}.upsamplers.0.conv"),
                c,
                c,
                3,
            );
        }
    }
    norm(&mut v, "decoder.conv_norm_out".into());
    conv(&mut v, "decoder.conv_out".into(), 3, c, 3);
    conv(&mut v, "quant_conv".into(), 2 * c, 2 * c, 1);
    conv(&mut v, "post_quant_conv".into(), c, c, 1);
    v.push(("bn.running_mean".into(), vec![4 * c], true));
    v.push(("bn.running_var".into(), vec![4 * c], true));
    v
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .map(|x| (bf16(*x).to_bits() >> 16) as u16)
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// flux2-mini's weights as a diffusers pipeline's checkpoint, in `dir/model.safetensors`: the
/// DiT under `dit.` (the poem's read rules inverted), the text encoder under `te.` (bf16), the
/// VAE under `vae.` ([`vae_names`], [`vae_plane`]).
pub fn checkpoint(trace: &Trace, dir: &Path) {
    let shape = |n: &str| -> Vec<usize> {
        let p = (trace.params.iter())
            .find(|p| p.name == n)
            .unwrap_or_else(|| panic!("flux2-mini has no param {n}"));
        p.shape.iter().map(|&d| d as usize).collect()
    };
    let p = |n: &str| -> Plane {
        let s = shape(n);
        let v = plane(trace, n, s.iter().product());
        (s, v)
    };
    let dim = 256usize;
    // rows `from..to` of a [rows, width] plane
    let rows_of = |(s, v): &Plane, from: usize, to: usize| {
        let w: usize = s[1..].iter().product();
        let mut s2 = s.clone();
        s2[0] = to - from;
        (s2, v[from * w..to * w].to_vec())
    };
    let adaln = |n: &str, order: &[usize]| {
        let whole = p(n);
        let w = whole.0[1];
        let mut v = Vec::new();
        for &i in order {
            v.extend_from_slice(&whole.1[i * dim * w..(i + 1) * dim * w]);
        }
        (whole.0.clone(), v)
    };
    let mut sd: Vec<(String, Plane)> = vec![
        (
            "time_guidance_embed.timestep_embedder.linear_1.weight".into(),
            p("dit.t_embed.linear_1"),
        ),
        (
            "time_guidance_embed.timestep_embedder.linear_2.weight".into(),
            p("dit.t_embed.linear_2"),
        ),
        (
            "time_guidance_embed.guidance_embedder.linear_1.weight".into(),
            p("dit.g_embed.linear_1"),
        ),
        (
            "time_guidance_embed.guidance_embedder.linear_2.weight".into(),
            p("dit.g_embed.linear_2"),
        ),
        (
            "double_stream_modulation_img.linear.weight".into(),
            adaln("dit.mod_img", &[1, 0, 2, 4, 3, 5]),
        ),
        (
            "double_stream_modulation_txt.linear.weight".into(),
            adaln("dit.mod_txt", &[1, 0, 2, 4, 3, 5]),
        ),
        (
            "single_stream_modulation.linear.weight".into(),
            adaln("dit.mod_single", &[1, 0, 2]),
        ),
        ("norm_out.linear.weight".into(), p("dit.norm_out")),
        ("context_embedder.weight".into(), p("dit.context_embed")),
        ("x_embedder.weight".into(), p("dit.x_embed")),
        ("proj_out.weight".into(), p("dit.proj_out")),
    ];
    for i in 0..2 {
        let b = format!("transformer_blocks.{i}");
        for (side, qkv, norms, out, ff) in [
            (
                "img",
                ["to_q", "to_k", "to_v"],
                ["norm_q", "norm_k"],
                "to_out.0",
                "ff",
            ),
            (
                "txt",
                ["add_q_proj", "add_k_proj", "add_v_proj"],
                ["norm_added_q", "norm_added_k"],
                "to_add_out",
                "ff_context",
            ),
        ] {
            let s = format!("dit.double.{i}.{side}");
            let fused = p(&format!("{s}.attn.qkv"));
            for (j, n) in qkv.iter().enumerate() {
                sd.push((
                    format!("{b}.attn.{n}.weight"),
                    rows_of(&fused, j * dim, (j + 1) * dim),
                ));
            }
            sd.push((
                format!("{b}.attn.{}.weight", norms[0]),
                p(&format!("{s}.attn.q_norm")),
            ));
            sd.push((
                format!("{b}.attn.{}.weight", norms[1]),
                p(&format!("{s}.attn.k_norm")),
            ));
            sd.push((
                format!("{b}.attn.{out}.weight"),
                p(&format!("{s}.attn.out")),
            ));
            sd.push((
                format!("{b}.{ff}.linear_in.weight"),
                p(&format!("{s}.ff.linear_in")),
            ));
            sd.push((
                format!("{b}.{ff}.linear_out.weight"),
                p(&format!("{s}.ff.linear_out")),
            ));
        }
    }
    for i in 0..2 {
        let (b, s) = (
            format!("single_transformer_blocks.{i}.attn"),
            format!("dit.single.{i}"),
        );
        sd.push((
            format!("{b}.to_qkv_mlp_proj.weight"),
            p(&format!("{s}.in_proj")),
        ));
        sd.push((format!("{b}.norm_q.weight"), p(&format!("{s}.q_norm"))));
        sd.push((format!("{b}.norm_k.weight"), p(&format!("{s}.k_norm"))));
        // to_out = [out_attn | out_mlp] along the input axis
        let (a, m) = (p(&format!("{s}.out_attn")), p(&format!("{s}.out_mlp")));
        let (wa, wm) = (a.0[1], m.0[1]);
        let mut v = Vec::new();
        for r in 0..a.0[0] {
            v.extend_from_slice(&a.1[r * wa..(r + 1) * wa]);
            v.extend_from_slice(&m.1[r * wm..(r + 1) * wm]);
        }
        sd.push((format!("{b}.to_out.weight"), (vec![a.0[0], wa + wm], v)));
    }
    let mut all: Vec<(String, Plane, bool)> = sd
        .into_iter()
        .map(|(n, p)| (format!("dit.{n}"), p, false))
        .collect();
    // the text encoder: Qwen3's names, the poem's fused planes split
    let q = 256usize;
    let mut te = vec![("te.model.embed_tokens.weight".to_string(), p("te.embed"))];
    for l in 0..2 {
        let (s, d) = (format!("te.layers.{l}"), format!("te.model.layers.{l}"));
        let qkv = p(&format!("{s}.qkv"));
        for (j, n) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
            te.push((
                format!("{d}.self_attn.{n}.weight"),
                rows_of(&qkv, j * q, (j + 1) * q),
            ));
        }
        te.push((
            format!("{d}.self_attn.q_norm.weight"),
            p(&format!("{s}.q_norm")),
        ));
        te.push((
            format!("{d}.self_attn.k_norm.weight"),
            p(&format!("{s}.k_norm")),
        ));
        te.push((format!("{d}.self_attn.o_proj.weight"), p(&format!("{s}.o"))));
        te.push((
            format!("{d}.input_layernorm.weight"),
            p(&format!("{s}.attn_norm")),
        ));
        te.push((
            format!("{d}.post_attention_layernorm.weight"),
            p(&format!("{s}.mlp_norm")),
        ));
        let gu = p(&format!("{s}.gate_up"));
        let half = gu.0[0] / 2;
        te.push((format!("{d}.mlp.gate_proj.weight"), rows_of(&gu, 0, half)));
        te.push((
            format!("{d}.mlp.up_proj.weight"),
            rows_of(&gu, half, 2 * half),
        ));
        te.push((format!("{d}.mlp.down_proj.weight"), p(&format!("{s}.down"))));
    }
    all.extend(te.into_iter().map(|(n, p)| (n, p, false)));
    for (n, shape, f32) in vae_names() {
        let v = vae_plane(&format!("vae.{n}"), &shape);
        all.push((format!("vae.{n}"), (shape, v), f32));
    }
    let mut entries = Vec::new();
    let mut data = Vec::new();
    for (name, (s, v), f32) in &all {
        let (dtype, bytes) = if *f32 {
            (
                "F32",
                v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
            )
        } else {
            ("BF16", bf16_bytes(v))
        };
        let dims: Vec<String> = s.iter().map(usize::to_string).collect();
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            dims.join(","),
            data.len(),
            data.len() + bytes.len()
        ));
        data.extend_from_slice(&bytes);
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    std::fs::create_dir_all(dir).expect("a checkpoint directory");
    std::fs::write(dir.join("model.safetensors"), out).expect("the checkpoint is written");
}

/// The import contract the runtime's `contract_for` makes of a catalog model's checkpoint.
pub fn contract_for(
    trace: &Trace,
    path: &Path,
) -> Result<checkpoint::contract::ModelContract, String> {
    poem_compiler::catalog::repository().contract_for(trace, path)
}

pub fn npy(rows: usize, width: usize, v: &[f32]) -> Vec<u8> {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {width}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend(v.iter().flat_map(|x| x.to_le_bytes()));
    out
}

/// Every node's values of the step, as the stream compile reads a capture
/// (`nodes.json`, `n<node>_{in,out}<k>.npy`; ints as floats).
pub fn capture(forward: &mut Forward, trace: &Trace, step: &StreamStep<'_>, dir: &Path) {
    let all: Vec<usize> = (0..trace.nodes.len()).collect();
    let (_, caps) = forward.stream_capture(step, &all).expect("the step runs");
    std::fs::create_dir_all(dir).expect("a capture directory");
    let mut index = Vec::new();
    for &node in &all {
        let (mut ins, mut outs) = (0, 0);
        let mut files = Vec::new();
        for c in caps.iter().filter(|c| c.node == node) {
            let k = if c.input { &mut ins } else { &mut outs };
            let name = format!("n{node}_{}{k}.npy", if c.input { "in" } else { "out" });
            *k += 1;
            let v: Vec<f32> = match &c.values {
                Values::F32(v) => v.clone(),
                Values::I32(v) => v.iter().map(|&x| x as f32).collect(),
            };
            std::fs::write(dir.join(&name), npy(c.rows, c.width, &v)).expect("a capture file");
            files.push(format!("\"{name}\": {}", c.value.0));
        }
        if !files.is_empty() {
            index.push(format!(
                "\"{node}\": {{\"op\": \"{}\", \"values\": {{{}}}}}",
                trace.nodes[node].op.name(),
                files.join(", ")
            ));
        }
    }
    std::fs::write(
        dir.join("nodes.json"),
        format!("{{{}}}\n", index.join(",\n")),
    )
    .expect("the capture's index");
}

/// The relative L2 distance of `a` from `b`, and their cosine.
pub fn rel_cos(a: &[f32], b: &[f32]) -> (f64, f64) {
    let dot = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(p, q)| f64::from(*p) * f64::from(*q))
            .sum::<f64>()
    };
    let d: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    (
        (dot(&d, &d) / dot(b, b)).sqrt(),
        dot(a, b) / (dot(a, a) * dot(b, b)).sqrt(),
    )
}
