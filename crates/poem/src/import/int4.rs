//! AWQ and GPTQ int4 projections, landed as the affine codes they are.
//!
//! A group of AWQ or GPTQ codes is `w = scale * (q - zero)`, one scale and one
//! zero a group of input channels: an affine weight, `w = scale * q + bias`
//! with `bias = -scale * zero`. A group of 128 is two groups of 64 under the
//! same factors, so a projection lands as `u4g64` (or `u4g32`) with its codes
//! moved, not requantized: only its fp16 factors are rounded, once, to bf16.
//!
//! AWQ (`quant_method = "awq"`, GEMM layout) stores `qweight` `[in, out / 8]`,
//! eight output channels a word in the nibble order 0, 2, 4, 6, 1, 3, 5, 7;
//! `qzeros` `[in / g, out / 8]` the same way; `scales` `[in / g, out]`.
//! GPTQ stores `qweight` `[in / 8, out]`, eight input channels a word, the low
//! nibble first; `qzeros` `[in / g, out / 8]` in order; `scales` as AWQ; and
//! `g_idx`, each input channel's group. Its first export (`checkpoint_format`
//! `gptq`, or none stated) stores each zero less one; `gptq_v2` stores it as
//! it is. An act-order export, whose `g_idx` is not each channel's own run of
//! `g`, is refused: its groups are not runs of the contracted axis.

use checkpoint::contract::{Expr, TensorContract, TensorType};
use checkpoint::file::snapshot::CONFIG;
use checkpoint::types::{DType, Encoding, RepackLayout};
use ztensor::format::cbor::Value;

use super::{
    Error, divided, encoding, extents, group_of, grouped, holds_the_declared_rectangle, joined,
    pack_axis, placed_from_encode, placed_road_covers, scaling,
};
use crate::{Dtype, Weight};

/// The two int4 exports, as a checkpoint's projections are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scheme {
    Awq,
    /// GPTQ, its zeros stored less one when `shifted`.
    Gptq {
        shifted: bool,
    },
}

/// Whether the projection named `part` (`{stem}.weight`) is stored as AWQ or
/// GPTQ int4: no `.weight`, a `.qweight` beside it.
pub(super) fn stored_int4(src: &ztensor::Source, part: &str) -> bool {
    part.strip_suffix(".weight")
        .is_some_and(|stem| src.get(part).is_none() && src.get(&qweight(stem)).is_some())
}

/// `w` landed from the AWQ or GPTQ projections `parts`, joined along `w`'s
/// cut axis when there are several.
pub(super) fn int4_planes(
    src: &ztensor::Source,
    w: &Weight,
    parts: &[String],
) -> Result<Vec<TensorContract>, Error> {
    if placed_road_covers(w) {
        return placed_from_encode(w, |canon| int4_planes(src, canon, parts));
    }
    let refuse = |detail: String| Error::Illegible {
        name: w.name.clone(),
        detail,
    };
    let scheme = scheme(src)?;
    if !matches!(w.dtype, Dtype::U4g64 | Dtype::U4g32) {
        return Err(refuse(format!(
            "`{}` is stored as {} int4 codes, and this deployment reads it as {:?}; \
             import the deployment that lands it as u4g64",
            parts[0],
            scheme_name(scheme),
            w.dtype
        )));
    }
    let declared = extents(w);
    let [_, k] = declared[..] else {
        return Err(refuse(format!(
            "it is declared {declared:?}, and an int4 projection is a [out, in] matrix"
        )));
    };
    let pairing = scaling(w);
    if pairing.channel_axis != 1 {
        return Err(refuse(format!(
            "its groups run along axis {}, and an int4 projection groups its input \
             channels, axis 1",
            pairing.channel_axis
        )));
    }
    let target = i64::from(group_of(w.dtype));
    let axis = if parts.len() > 1 { pack_axis(w) } else { 0 };
    let f32 = Encoding::Raw(DType::F32);
    let mut out = Vec::new();
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    let mut legs = Vec::new();
    let mut stored_factors = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        let stem = part.strip_suffix(".weight").ok_or_else(|| {
            refuse(format!(
                "`{part}` names no `.weight`, beside which int4 planes are named"
            ))
        })?;
        let (words, zeros, factors) = (
            qweight(stem),
            format!("{stem}.qzeros"),
            format!("{stem}.scales"),
        );
        let shape = |name: &str| -> Result<[i64; 2], Error> {
            let tensor = src
                .get(name)
                .ok_or_else(|| Error::Missing(name.to_string()))?;
            match tensor.shape() {
                &[a, b] => Ok([a, b].map(|n| i64::try_from(n).expect("an extent inside i64"))),
                other => Err(refuse(format!(
                    "`{name}` is stored {other:?}, not a matrix"
                ))),
            }
        };
        let raw = |name: &str, want: &[DType]| -> Result<DType, Error> {
            match super::stored_encoding(src, name)? {
                Encoding::Raw(dtype) if want.contains(&dtype) => Ok(dtype),
                other => Err(refuse(format!(
                    "`{name}` is stored {other:?}, and {} stores it as {want:?}",
                    scheme_name(scheme)
                ))),
            }
        };
        raw(&words, &[DType::I32, DType::U32])?;
        raw(&zeros, &[DType::I32, DType::U32])?;
        let factor = raw(&factors, &[DType::F16, DType::Bf16, DType::F32])?;
        stored_factors.push(factor);
        let [a, b] = shape(&words)?;
        let (inputs, rows, code_layout, zero_layout) = match scheme {
            Scheme::Awq => (a, b * 8, RepackLayout::AwqU4Codes, RepackLayout::AwqU4Zeros),
            Scheme::Gptq { .. } => (
                a * 8,
                b,
                RepackLayout::GptqU4Codes,
                RepackLayout::GptqU4Zeros,
            ),
        };
        if inputs != k {
            return Err(refuse(format!(
                "`{words}` is stored [{a}, {b}], {inputs} input channels as {} packs them, \
                 and this model reads {k}",
                scheme_name(scheme)
            )));
        }
        let [groups, outputs] = shape(&factors)?;
        if outputs != rows || groups <= 0 || k % groups != 0 || shape(&zeros)? != [groups, rows / 8]
        {
            return Err(refuse(format!(
                "`{factors}` is stored [{groups}, {outputs}] and `{zeros}` {:?}, which is \
                 not one scale and zero a group of `{words}`'s {rows} output channels",
                shape(&zeros)?
            )));
        }
        let group = k / groups;
        // A zero is scaled by its group's scale as a block of the landed
        // groups it spans, and a scale blocks two or more: a group as narrow
        // as a landed one would be an elementwise product.
        if group % target != 0 || group == target {
            return Err(refuse(format!(
                "`{words}` groups {group} input channels, and {:?} lands groups of \
                 {target}: an int4 group lands as two or more of them",
                w.dtype
            )));
        }
        if let Some(order) = src.get(&format!("{stem}.g_idx")) {
            in_runs(&order, &format!("{stem}.g_idx"), k, group).map_err(refuse)?;
        }
        legs.push(vec![rows, k]);
        let leg = |what: &str| format!("{}.int4.{i}.{what}", w.name);
        let landed: Vec<i64> = (0..k / target).map(|j| j * target / group).collect();
        let (stored, counted) = (vec![rows, groups], vec![rows, k / target]);
        let staged = |what: &str, expr: Expr, shape: &Vec<i64>, encoding: &Encoding| {
            TensorContract::new(leg(what), expr, shape.clone(), encoding.clone()).internal()
        };
        let factor = Encoding::Raw(factor);
        let bytes = Encoding::Raw(DType::U8);
        out.extend([
            TensorContract::inferred(
                leg("codes"),
                Expr::src(words.clone())
                    .repack(code_layout, TensorType::new(vec![rows, k], grouped(w))),
                grouped(w),
            )
            .internal(),
            staged(
                "scales",
                Expr::src(factors.clone()).repack(
                    RepackLayout::Transposed,
                    TensorType::new(stored.clone(), factor.clone()),
                ),
                &stored,
                &factor,
            ),
            staged(
                "zeros",
                Expr::src(zeros.clone())
                    .repack(zero_layout, TensorType::new(stored.clone(), bytes.clone())),
                &stored,
                &bytes,
            ),
            staged(
                "zeros.f32",
                Expr::out(leg("zeros")).cast(f32.clone()),
                &stored,
                &f32,
            ),
            staged(
                "zeros.landed",
                Expr::out(leg("zeros.f32")).gather(1, landed.clone()),
                &counted,
                &f32,
            ),
            staged(
                "scales.landed",
                Expr::out(leg("scales")).gather(1, landed),
                &counted,
                &factor,
            ),
        ]);
        // `-zero`, and GPTQ's first export's `-(stored + 1)`.
        out.push(staged(
            "zeros.negated",
            Expr::out(leg("zeros.landed")).scale(-1.0),
            &counted,
            &f32,
        ));
        let negated = if scheme == (Scheme::Gptq { shifted: true }) {
            out.push(staged(
                "zeros.unshifted",
                Expr::out(leg("zeros.negated")).bias(-1.0),
                &counted,
                &f32,
            ));
            leg("zeros.unshifted")
        } else {
            leg("zeros.negated")
        };
        out.push(staged(
            "biases.landed",
            Expr::out(negated).scale_per_block(Expr::out(leg("scales"))),
            &counted,
            &f32,
        ));
        scales.push(Expr::out(leg("scales.landed")));
        biases.push(Expr::out(leg("biases.landed")));
        codes.push(Expr::out(leg("codes")));
    }
    holds_the_declared_rectangle(w, axis, &legs)?;
    let counted = divided(&declared, pairing.channel_axis, pairing.group_size, &w.name);
    let bf16 = encoding(Dtype::Bf16);
    out.push(TensorContract::inferred(
        w.name.clone(),
        joined(axis, codes),
        grouped(w),
    ));
    out.push(
        TensorContract::new(
            crate::scales_name(&w.name),
            // Scales stored bf16 land as they are.
            if stored_factors.iter().all(|&f| f == DType::Bf16) {
                joined(axis, scales)
            } else {
                joined(axis, scales).cast(bf16.clone())
            },
            counted.clone(),
            bf16.clone(),
        )
        .scaling(pairing),
    );
    out.push(
        TensorContract::new(
            crate::biases_name(&w.name),
            joined(axis, biases).cast(bf16.clone()),
            counted,
            bf16,
        )
        .offsetting(w.name.clone()),
    );
    Ok(out)
}

fn qweight(stem: &str) -> String {
    format!("{stem}.qweight")
}

fn scheme_name(scheme: Scheme) -> &'static str {
    match scheme {
        Scheme::Awq => "AWQ",
        Scheme::Gptq { .. } => "GPTQ",
    }
}

/// The int4 export the checkpoint's `quantization_config` states, refusing
/// what it states that no plane here lands.
fn scheme(src: &ztensor::Source) -> Result<Scheme, Error> {
    let config = src
        .attributes()
        .and_then(|a| a.get(CONFIG))
        .and_then(|c| c.get("quantization_config"));
    let text = |key: &str| config.and_then(|c| c.get(key)).and_then(Value::as_text);
    let refuse = |detail: String| Error::Illegible {
        name: "quantization_config".to_string(),
        detail,
    };
    if let Some(bits) = config.and_then(|c| c.get("bits")).and_then(Value::as_u64)
        && bits != 4
    {
        return Err(refuse(format!(
            "it states {bits}-bit codes, and an int4 projection lands 4"
        )));
    }
    match text("quant_method") {
        Some("awq") => match text("version") {
            None | Some("gemm" | "GEMM") => Ok(Scheme::Awq),
            Some(other) => Err(refuse(format!(
                "it states AWQ's `{other}` layout, and AWQ lands from its GEMM layout"
            ))),
        },
        Some("gptq") => match text("checkpoint_format") {
            None | Some("gptq") => Ok(Scheme::Gptq { shifted: true }),
            Some("gptq_v2") => Ok(Scheme::Gptq { shifted: false }),
            Some(other) => Err(refuse(format!(
                "it states GPTQ's `{other}` checkpoint format, and GPTQ lands from \
                 `gptq` or `gptq_v2`"
            ))),
        },
        Some(other) => Err(refuse(format!(
            "it states `{other}`, and a `.qweight` lands as AWQ or GPTQ int4"
        ))),
        None => Err(refuse(
            "it states no `quant_method`, and a `.qweight` is AWQ's or GPTQ's layout by it"
                .to_string(),
        )),
    }
}

/// GPTQ's `g_idx`: every input channel in its own run of `group`, not an
/// act-order permutation.
fn in_runs(order: &ztensor::Tensor<'_>, name: &str, k: i64, group: i64) -> Result<(), String> {
    let bytes = order
        .bytes()
        .map_err(|why| format!("`{name}` cannot be read ({why})"))?;
    let groups: Vec<i64> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i64::from(i32::from_le_bytes(*c)))
        .collect();
    if i64::try_from(groups.len()).ok() != Some(k) {
        return Err(format!(
            "`{name}` holds {} groups, one an input channel, and this model reads {k}",
            groups.len()
        ));
    }
    match (0..k).zip(&groups).find(|(c, g)| c / group != **g) {
        None => Ok(()),
        Some((c, g)) => Err(format!(
            "`{name}` puts input channel {c} in group {g}, not {}: an act-order \
             (desc_act) export, whose groups are not runs of the input channels; \
             export it without act order",
            c / group
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ALIGNMENT, planes, planes_fused};
    use super::*;
    use checkpoint::contract::ModelContract;
    use checkpoint::executor::Execution;
    use checkpoint::executor::sink::MemorySink;
    use checkpoint::plan::{CONVERT_TILE_MAP_MASK, StorageTarget, compile};

    const N: usize = 16;
    const K: usize = 256;
    const G: usize = 128;
    const AWQ_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

    /// One projection's codes `[out, in]`, zeros and scales `[out, in / G]`.
    struct Leg {
        q: Vec<Vec<u32>>,
        z: Vec<Vec<u32>>,
        s: Vec<Vec<f32>>,
    }

    /// A projection of codes, zeros 1..=15 and scales bf16 and f16 both
    /// hold exactly, so a landed weight is the reference to the bit.
    fn leg(seed: u32) -> Leg {
        let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let mut next = |m: u32| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 16) % m
        };
        let q = (0..N).map(|_| (0..K).map(|_| next(16)).collect()).collect();
        let z = (0..N)
            .map(|_| (0..K / G).map(|_| 1 + next(15)).collect())
            .collect();
        let s = (0..N)
            .map(|_| {
                (0..K / G)
                    .map(|_| (1.0 + next(8) as f32 / 8.0) / 128.0)
                    .collect()
            })
            .collect();
        Leg { q, z, s }
    }

    fn f16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        assert!(
            (1..31).contains(&exp) && bits & 0x1FFF == 0,
            "{v} is an exact f16"
        );
        (((bits >> 16) & 0x8000) | ((exp as u32) << 10) | ((bits >> 13) & 0x3FF)) as u16
    }

    fn words(word: impl Fn(usize, usize) -> u32, rows: usize, cols: usize) -> Vec<u8> {
        (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (r, c)))
            .flat_map(|(r, c)| word(r, c).to_le_bytes())
            .collect()
    }

    /// `l` as AWQ (GEMM) or GPTQ (v1, zeros less one) stores it under `stem`.
    fn write(
        writer: &mut ztensor::Writer,
        stem: &str,
        l: &Leg,
        awq: bool,
        g_idx: Option<Vec<i32>>,
    ) {
        let pack = |v: &dyn Fn(usize) -> u32, order: bool| {
            (0..8).fold(0u32, |w, j| {
                w | (v(if order { AWQ_ORDER[j] } else { j }) << (4 * j))
            })
        };
        let qweight = if awq {
            words(|r, c| pack(&|j| l.q[8 * c + j][r], true), K, N / 8)
        } else {
            words(|r, c| pack(&|j| l.q[c][8 * r + j], false), K / 8, N)
        };
        let shape = if awq { [K, N / 8] } else { [K / 8, N] };
        let zero = |o: usize, g: usize| if awq { l.z[o][g] } else { l.z[o][g] - 1 };
        let qzeros = words(|g, c| pack(&|j| zero(8 * c + j, g), awq), K / G, N / 8);
        let scales: Vec<u8> = (0..K / G)
            .flat_map(|g| (0..N).map(move |o| (o, g)))
            .flat_map(|(o, g)| {
                // AWQ stores its scales f16 here, GPTQ bf16: both land.
                let s = l.s[o][g];
                if awq {
                    f16_bits(s)
                } else {
                    (s.to_bits() >> 16) as u16
                }
                .to_le_bytes()
            })
            .collect();
        let at = |plane: &str| format!("{stem}.{plane}");
        let shape = shape.map(|n| n as u64).to_vec();
        if let Some(g_idx) = g_idx {
            let bytes: Vec<u8> = g_idx.iter().flat_map(|g| g.to_le_bytes()).collect();
            writer
                .add(at("g_idx"), vec![K as u64], ztensor::Leaf::I32, &bytes)
                .expect("g_idx lands");
        }
        writer
            .add(at("qweight"), shape, ztensor::Leaf::I32, &qweight)
            .expect("codes land");
        writer
            .add(
                at("qzeros"),
                vec![(K / G) as u64, (N / 8) as u64],
                ztensor::Leaf::I32,
                &qzeros,
            )
            .expect("zeros land");
        writer
            .add(
                at("scales"),
                vec![(K / G) as u64, N as u64],
                if awq {
                    ztensor::Leaf::F16
                } else {
                    ztensor::Leaf::BF16
                },
                &scales,
            )
            .expect("scales land");
    }

    fn source(
        dir: &std::path::Path,
        method: &str,
        legs: &[(&str, &Leg)],
        g_idx: Option<Vec<i32>>,
    ) -> ztensor::Source {
        let path = dir.join("model.zt");
        let mut writer = ztensor::Writer::create(&path).expect("the container opens");
        for (stem, l) in legs {
            write(&mut writer, stem, l, method == "awq", g_idx.clone());
        }
        writer.finish().expect("the container closes");
        let text = |s: &str| Value::Text(s.to_string());
        let config = Value::Map(vec![(
            text("quantization_config"),
            Value::Map(vec![
                (text("quant_method"), text(method)),
                (text("bits"), Value::Uint(4)),
                (text("group_size"), Value::Uint(G as u64)),
            ]),
        )]);
        ztensor::Source::open(&path)
            .expect("it reads back")
            .with_attribute(CONFIG, config)
    }

    /// The landed `u4g64` weight, dequantized: `scale * q + bias` a 64-group.
    fn landed(dir: &std::path::Path, read: Vec<TensorContract>, name: &str) -> Vec<Vec<f32>> {
        let contract = ModelContract {
            alignment: ALIGNMENT,
            tensors: read,
            groups: Vec::new(),
        };
        let metadata = checkpoint::file::read::parse_metadata(dir).expect("it parses");
        let plan = compile(
            &metadata,
            &contract,
            StorageTarget {
                tile_map_mask: CONVERT_TILE_MAP_MASK,
                ..StorageTarget::default()
            },
        )
        .expect("it compiles");
        let mut sink = MemorySink::default();
        Execution::new(&plan, dir)
            .sink(&mut sink)
            .run()
            .expect("the plan executes");
        let bf16s = |plane: &str| -> Vec<f32> {
            sink.tensors[plane]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
                .collect()
        };
        let (codes, scales, biases) = (
            &sink.tensors[name],
            bf16s(&crate::scales_name(name)),
            bf16s(&crate::biases_name(name)),
        );
        let rows = codes.len() * 2 / K;
        (0..rows)
            .map(|o| {
                (0..K)
                    .map(|i| {
                        let at = o * K + i;
                        let q = (codes[at / 2] >> (4 * (at % 2))) & 0xF;
                        let g = o * (K / 64) + i / 64;
                        scales[g] * f32::from(q) + biases[g]
                    })
                    .collect()
            })
            .collect()
    }

    fn reference(l: &Leg) -> Vec<Vec<f32>> {
        (0..N)
            .map(|o| {
                (0..K)
                    .map(|i| l.s[o][i / G] * (l.q[o][i] as f32 - l.z[o][i / G] as f32))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn int4_every_case() {
        awq_legs_land_as_u4g64_that_dequantize_to_awq();
        gptq_lands_as_u4g64_that_dequantizes_to_gptq_its_zeros_one_up();
        an_act_order_gptq_and_a_bf16_reading_are_refused_by_name();
    }

    fn awq_legs_land_as_u4g64_that_dequantize_to_awq() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let (gate, up) = (leg(1), leg(2));
        let src = source(
            dir.path(),
            "awq",
            &[("mlp.gate_proj", &gate), ("mlp.up_proj", &up)],
            None,
        );
        let w = Weight::sym("gate_up", [2 * N as u64, K as u64], Dtype::U4g64)
            .packed([N as u64, N as u64]);
        let parts = ["mlp.gate_proj.weight", "mlp.up_proj.weight"].map(String::from);
        let read = planes_fused(&src, &w, parts).expect("AWQ legs read as one u4g64 weight");
        let public: Vec<&str> = read
            .iter()
            .filter(|t| t.visibility == checkpoint::types::Visibility::Public)
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(public, ["gate_up", "gate_up.scales", "gate_up.biases"]);
        let got = landed(dir.path(), read, "gate_up");
        assert_eq!(
            got,
            [reference(&gate), reference(&up)].concat(),
            "AWQ lands to the bit"
        );
    }

    fn gptq_lands_as_u4g64_that_dequantizes_to_gptq_its_zeros_one_up() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let proj = leg(3);
        let runs = (0..K as i32).map(|i| i / G as i32).collect();
        let src = source(dir.path(), "gptq", &[("attn.o_proj", &proj)], Some(runs));
        let w = Weight::sym("o", [N as u64, K as u64], Dtype::U4g64);
        let read = planes(&src, &w, "attn.o_proj.weight").expect("GPTQ reads as u4g64");
        assert_eq!(
            landed(dir.path(), read, "o"),
            reference(&proj),
            "GPTQ lands to the bit"
        );
    }

    fn an_act_order_gptq_and_a_bf16_reading_are_refused_by_name() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let proj = leg(4);
        let mut shuffled: Vec<i32> = (0..K as i32).map(|i| i / G as i32).collect();
        shuffled.swap(0, K - 1);
        let src = source(
            dir.path(),
            "gptq",
            &[("attn.o_proj", &proj)],
            Some(shuffled),
        );
        let w = Weight::sym("o", [N as u64, K as u64], Dtype::U4g64);
        let why = planes(&src, &w, "attn.o_proj.weight")
            .expect_err("act order is refused")
            .to_string();
        assert!(why.contains("act-order"), "{why}");

        let dense = Weight::sym("o", [N as u64, K as u64], Dtype::Bf16);
        let why = planes(&src, &dense, "attn.o_proj.weight")
            .expect_err("bf16 is refused")
            .to_string();
        assert!(why.contains("u4g64"), "{why}");
    }
}
