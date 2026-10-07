//! A catalog SKU end to end on the device, over random weights: writes a
//! checkpoint of every plan param at small random values (bf16 of the
//! logical shape for a quantized bank, which the landing quantizes: affine
//! u4g64 and mxfp4 have encoders), lands it, then
//!
//! - fires every class of the plan at a prefill and a decode shape
//!   (`Shell::synthetic_fires`) and checks what comes back is finite;
//! - for a plan that reads logits: prefills a random prompt on one slot and
//!   walks the same prompt one token at a time on another, and compares the
//!   last rows (argmax, max |Δ|, correlation).
//!
//! ```text
//! PIE_XLA_PLUGIN=.../libtpu.so cargo run --release -p engine-xla --example tiny_e2e -- <sku> [--keep]
//! ```
//!
//! The checkpoint goes to `/dev/shm/pie-xla-e2e/<sku>.zt` (removed unless
//! `--keep`). The device lock is held only while the shell is up.

use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_xla::DeviceBoot;
use engine_xla::serve::{Boot, Lane, Shell};
use poem_compiler::Budget;
use poem_dsl::{Dtype, ParamSource, Platform, Request, Weight};

fn main() {
    let mut args = std::env::args().skip(1);
    let name = args.next().expect("usage: tiny_e2e <sku> [--keep]");
    let keep = args.any(|a| a == "--keep");
    let sku = models::deployment(&name)
        .or_else(|| models::deployments().find(|s| s.name.starts_with(&name)))
        .unwrap_or_else(|| panic!("no SKU `{name}`"));
    let trace = sku.trace(Platform::Xla);
    let dir = PathBuf::from("/dev/shm/pie-xla-e2e");
    std::fs::create_dir_all(&dir).expect("the scratch directory");
    let path = dir.join(format!("{}.zt", sku.name));
    let t0 = Instant::now();
    let bytes = write_random(&trace, &path);
    eprintln!(
        "{}: wrote {} MiB of random weights in {:.1}s",
        sku.name,
        bytes >> 20,
        t0.elapsed().as_secs_f64()
    );
    if std::env::var_os("PIE_E2E_WRITE_ONLY").is_some() {
        let source = ztensor_compat::index(&path).expect("the checkpoint indexes");
        checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Xla)
            .unwrap_or_else(|why| panic!("the random checkpoint reads: {why}"));
        println!("{}: written and read", sku.name);
        return;
    }
    let source = ztensor_compat::index(&path).expect("the checkpoint indexes");
    let contract = checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Xla)
        .unwrap_or_else(|why| panic!("the random checkpoint reads: {why}"));
    drop(source);

    let outcome = run(sku, trace, &contract, &path);
    if !keep {
        let _ = std::fs::remove_file(&path);
    }
    match outcome {
        Ok(line) => println!("{:<50} OK      {line}", sku.name),
        Err(why) => {
            println!("{:<50} FAILED  {why}", sku.name);
            std::process::exit(1);
        }
    }
}

fn run(
    sku: &models::Deployment,
    trace: poem_dsl::Trace,
    contract: &checkpoint::contract::ModelContract,
    path: &Path,
) -> Result<String, String> {
    let context = 512;
    let budgets = engine::load::Budgets {
        max_lanes: 4,
        max_tokens: 64,
        page_size: 16,
        max_context: context,
        slots: 4,
        pages: 4 * context / 16,
        ..engine::load::Budgets::default()
    };
    let patches = engine_xla::api::patch_ladder(&trace, &budgets);
    let voxels = engine_xla::dit::voxel_ladder(&trace, Some(4096), Some(4), budgets.max_lanes);
    let _device = engine_xla::bench::lock_device();
    let t0 = Instant::now();
    let mut shell = Shell::load_with(
        Boot {
            trace,
            contract,
            checkpoint: path,
            budget: Budget::new(budgets.max_lanes, budgets.max_tokens),
            patches,
            page_size: budgets.page_size,
            context,
            slots: budgets.slots,
            pages: budgets.pages,
            device: &DeviceBoot::default(),
        },
        voxels,
    )
    .map_err(|fault| format!("load: {fault}"))?;
    let loaded = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let probes = shell.synthetic_fires(24, true);
    let fired = t1.elapsed().as_secs_f64();
    let mut fires = 0;
    for p in &probes {
        match (&p.outcome, p.finite) {
            (Err(why), _) => {
                return Err(format!(
                    "class {} rows {}: {why} ({})",
                    p.class, p.rows, p.request
                ));
            }
            (Ok(_), Some((false, n))) => {
                return Err(format!(
                    "class {} rows {}: {n} values read back, not all finite ({})",
                    p.class, p.rows, p.request
                ));
            }
            _ => fires += 1,
        }
    }
    let mut line = format!("load {loaded:.1}s, {fires} synthetic fires finite ({fired:.1}s)");
    if shell.readout_seam() == engine::fire::ReadoutSeam::Logits {
        match agree(&mut shell, sku) {
            Ok(said) => line.push_str(&format!("; {said}")),
            // A plan whose token lanes read float ports (positions, a
            // timestep) has no plain prompt to walk; its synthetic fires
            // above fed them.
            Err(why) if why.contains("feeds it no channel") => {
                line.push_str("; its token lanes read ports, no prompt walk");
            }
            Err(why) => return Err(why),
        }
    }
    Ok(line)
}

/// Prefill against a token-by-token walk of the same prompt.
fn agree(shell: &mut Shell, sku: &models::Deployment) -> Result<String, String> {
    let facts = sku.trace(models::Platform::Xla).facts;
    let word = |query_len: u32| facts.word(&Request::new(query_len, false));
    let vocab = shell.out_width();
    let mut lcg = 0x2545_f491_4f6c_dd1du64;
    let prompt: Vec<u32> = (0..12)
        .map(|_| {
            lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((lcg >> 33) % u64::from(vocab.clamp(2, 1000))) as u32
        })
        .collect();
    let fault = |what: &'static str| move |f: engine_xla::Fault| format!("{what}: {f}");
    // `PIE_E2E_PROBE=1`: every token-rowed op output of the walk is read
    // back, and the first whose prefill row and walked row part is named.
    let probing = std::env::var_os("PIE_E2E_PROBE").is_some();
    if probing {
        let trace = shell.trace();
        let values: Vec<poem_ir::ValueId> = trace
            .values
            .iter()
            .enumerate()
            .filter(|(_, d)| {
                matches!(d.def, poem_ir::Def::Op(_))
                    && matches!(&d.ty, poem_ir::Ty::Tensor { shape, .. } if shape.first() == Some(&poem_ir::Dim::Tokens))
            })
            .map(|(at, _)| poem_ir::ValueId(at as u32))
            .collect();
        shell.probe(values);
    }
    shell.open(0).map_err(fault("open"))?;
    let first_probes = if probing {
        let seated = [engine_xla::serve::Seated::of(Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: &prompt,
        })];
        let fired = shell.fire_full(&seated).map_err(fault("probe prefill"))?;
        shell.open(0).map_err(fault("open"))?;
        fired.probes
    } else {
        Vec::new()
    };
    let first = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: &prompt,
        }])
        .map_err(fault("prefill"))?
        .remove(0);
    shell.open(1).map_err(fault("open"))?;
    let mut last = Vec::new();
    if probing {
        for id in &prompt[..prompt.len() - 1] {
            shell
                .fire(&[Lane {
                    slot: 1,
                    word: word(1),
                    tokens: &[*id],
                }])
                .map_err(fault("a decode"))?;
        }
        let seated = [engine_xla::serve::Seated::of(Lane {
            slot: 1,
            word: word(1),
            tokens: &prompt[prompt.len() - 1..],
        })];
        let fired = shell.fire_full(&seated).map_err(fault("probe decode"))?;
        let row = prompt.len() - 1;
        let mut shown = 0;
        for (value, width, plane) in &fired.probes {
            let Some((_, pw, pplane)) = first_probes.iter().find(|(v, _, _)| v == value) else {
                continue;
            };
            let (w, pw) = (*width as usize, *pw as usize);
            if w != pw || pplane.len() < (row + 1) * w || plane.len() < w {
                continue;
            }
            let a = &pplane[row * w..(row + 1) * w];
            let b = &plane[..w];
            let c = correlation(a, b);
            let node = match shell.trace().values[value.0 as usize].def {
                poem_ir::Def::Op(n) => n as usize,
                _ => 0,
            };
            let op = poem_ir::Operands::name(&shell.trace().nodes[node].op);
            if c < 0.999 && shown < 12 {
                eprintln!(
                    "  probe: value {} (node {node} `{op}`, layer {:?}): corr {c:.6}",
                    value.0,
                    shell.trace().nodes[node].layer
                );
                shown += 1;
            }
        }
        shell.open(1).map_err(fault("open"))?;
    }
    for id in &prompt {
        last = shell
            .fire(&[Lane {
                slot: 1,
                word: word(1),
                tokens: &[*id],
            }])
            .map_err(fault("a decode"))?
            .remove(0);
    }
    if first.is_empty() || first.len() != last.len() {
        return Err(format!(
            "readouts of {} and {} values",
            first.len(),
            last.len()
        ));
    }
    if !first.iter().chain(&last).all(|v| v.is_finite()) {
        return Err("non-finite logits".to_string());
    }
    let worst = first
        .iter()
        .zip(&last)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = first.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let corr = correlation(&first, &last);
    let (a, b) = (argmax(&first), argmax(&last));
    let line = format!(
        "prefill vs token-by-token: argmax {a} / {b}, max |Δ| {worst:.3e} of max |logit| {scale:.3e}, corr {corr:.6}"
    );
    if corr < 0.999 {
        return Err(line);
    }
    Ok(line)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let mb = b.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x) - ma, f64::from(y) - mb);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa == 0.0 || bb == 0.0 {
        return if aa == bb { 1.0 } else { 0.0 };
    }
    ab / (aa * bb).sqrt()
}

/// Every checkpoint param at small random values; answers the bytes written.
fn write_random(trace: &poem_dsl::Trace, path: &Path) -> u64 {
    let quantized = |d: Dtype| {
        matches!(
            d,
            Dtype::Mxfp4
                | Dtype::U4g64
                | Dtype::U8g64
                | Dtype::U4g32
                | Dtype::U2g32
                | Dtype::U2g64
                | Dtype::U2g128
        )
    };
    let companions: std::collections::BTreeSet<String> = trace
        .params
        .iter()
        .filter(|p| quantized(p.dtype))
        .flat_map(|p| [models::scales_name(&p.name), models::biases_name(&p.name)])
        .collect();
    let mut writer = ztensor::Writer::create(path).expect("the checkpoint opens");
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut total = 0u64;
    // A canonical container takes its tensors in name order.
    let mut params: Vec<&poem_dsl::Param> = trace.params.iter().collect();
    params.sort_by(|a, b| a.name.cmp(&b.name));
    for param in params {
        if param.source != ParamSource::Checkpoint || companions.contains(&param.name) {
            continue;
        }
        let w = Weight::of_plane(param);
        let shape: Vec<u64> = w.shape.clone();
        let n: u64 = shape.iter().product();
        let fan = shape.last().copied().unwrap_or(1).max(1) as f32;
        let scale = if shape.len() <= 1 {
            0.1
        } else {
            1.0 / fan.sqrt()
        };
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        };
        if quantized(param.dtype) {
            // A bank lands as one object of its canonical type: random codes,
            // small positive gains, small offsets.
            let spelled = serde_json::to_value(param.dtype).expect("a dtype spells");
            let term = ztensor::Term::parse(spelled.as_str().expect("a dtype spells as text"))
                .unwrap_or_else(|why| panic!("`{}`'s type parses: {why}", param.name));
            let planes = term.planes(&shape).expect("the type lays out its planes");
            if std::env::var_os("PIE_E2E_WRITE_ONLY").is_some() {
                eprintln!(
                    "  {} {spelled}: {:?}",
                    param.name,
                    planes.iter().map(|p| (&p.path, p.leaf)).collect::<Vec<_>>()
                );
            }
            let mut bufs: Vec<Vec<u8>> = Vec::new();
            for plane in &planes {
                let len = plane.len as usize;
                let elems = plane.elements() as usize;
                let gain = plane.path == "gain";
                let buf: Vec<u8> = match plane.leaf {
                    _ if plane.path == "code" => {
                        (0..len).map(|_| (next().to_bits() >> 3) as u8).collect()
                    }
                    ztensor::Leaf::BF16 => (0..elems)
                        .flat_map(|_| {
                            let v = if gain {
                                next().abs() * 0.05 + 0.01
                            } else {
                                next() * 0.05
                            };
                            ((v.to_bits() >> 16) as u16).to_le_bytes()
                        })
                        .collect(),
                    ztensor::Leaf::F16 => (0..elems)
                        .flat_map(|_| {
                            let v = if gain {
                                next().abs() * 0.05 + 0.01
                            } else {
                                next() * 0.05
                            };
                            kernels_xla::hlo::f16_bits(v).to_le_bytes()
                        })
                        .collect(),
                    ztensor::Leaf::F32 => (0..elems)
                        .flat_map(|_| {
                            let v = if gain {
                                next().abs() * 0.05 + 0.01
                            } else {
                                next() * 0.05
                            };
                            v.to_le_bytes()
                        })
                        .collect(),
                    // e8m0 exponents around 2^-5.
                    _ => (0..len).map(|i| 120 + (i % 5) as u8).collect(),
                };
                assert_eq!(buf.len(), len, "`{}` plane {}", param.name, plane.path);
                bufs.push(buf);
            }
            total += bufs.iter().map(|b| b.len() as u64).sum::<u64>();
            let slices: Vec<&[u8]> = bufs.iter().map(Vec::as_slice).collect();
            writer
                .object(param.name.clone(), |o| {
                    o.shape(shape.clone()).term(term.clone()).planes(slices)
                })
                .unwrap_or_else(|why| panic!("`{}` writes: {why}", param.name));
            continue;
        }
        let dtype = param.dtype;
        let (leaf, bytes): (ztensor::Leaf, Vec<u8>) = match dtype {
            Dtype::Bf16 => (
                ztensor::Leaf::BF16,
                (0..n)
                    .flat_map(|_| ((next().to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
            ),
            Dtype::F32 => (
                ztensor::Leaf::F32,
                (0..n).flat_map(|_| next().to_le_bytes()).collect(),
            ),
            Dtype::F16 => (
                ztensor::Leaf::F16,
                (0..n)
                    .flat_map(|_| kernels_xla::hlo::f16_bits(next()).to_le_bytes())
                    .collect(),
            ),
            Dtype::I32 => (ztensor::Leaf::I32, vec![0u8; n as usize * 4]),
            Dtype::I64 => (ztensor::Leaf::I64, vec![0u8; n as usize * 8]),
            Dtype::U8 | Dtype::Bool => (ztensor::Leaf::U8, vec![0u8; n as usize]),
            other => panic!(
                "`{}` is {other:?}, which this writer does not draw",
                param.name
            ),
        };
        total += bytes.len() as u64;
        writer
            .add(&param.name, shape, leaf, &bytes)
            .unwrap_or_else(|why| panic!("`{}` writes: {why}", param.name));
    }
    writer.finish().expect("the checkpoint closes");
    total
}
