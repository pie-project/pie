//! flux2-mini as engine-cpu's flux test and `mk_tiny_flux2.py` draw it: its weights and a step's
//! inputs, by name, at `READING_ROWS` rows a lane (default 64; a square latent of 2 x 2
//! patches), and its weights as a diffusers checkpoint.

#![allow(dead_code)]

use std::path::Path;

use poem_ir::Trace;

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

/// `count` values in [-1/2, 1/2), each exact in f32: what mk_tiny_flux2.py draws for `name`.
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

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .map(|x| (bf16(*x).to_bits() >> 16) as u16)
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// flux2-mini's weights as diffusers' `Flux2Transformer2DModel` state dict names and lays them
/// (`mk_tiny_flux2.py`'s `state_dict`: the poem's read rules inverted: q / k / v split, the adaLN
/// projections' (shift, scale) pairs swapped, a single block's two output projections side by
/// side), bf16, in `dir/diffusion_pytorch_model.safetensors`.
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
    let mut entries = Vec::new();
    let mut data = Vec::new();
    for (name, (s, v)) in &sd {
        let bytes = bf16_bytes(v);
        let dims: Vec<String> = s.iter().map(usize::to_string).collect();
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
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
    std::fs::write(dir.join("diffusion_pytorch_model.safetensors"), out)
        .expect("the checkpoint is written");
}

/// The import contract the runtime's `contract_for` makes of a catalog model's checkpoint.
pub fn contract_for(
    trace: &Trace,
    path: &Path,
) -> Result<checkpoint::contract::ModelContract, String> {
    let src = checkpoint::file::snapshot::open(path).map_err(|e| e.to_string())?;
    poem_compiler::catalog::deployment(&trace.name)
        .ok_or_else(|| format!("no catalog deployment {}", trace.name))?
        .contract(&src, trace.platform)
        .map_err(|e| format!("{e:?}"))
}
