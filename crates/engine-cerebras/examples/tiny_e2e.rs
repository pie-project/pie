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
//! cargo run --release -p engine-cerebras --example tiny_e2e -- <sku> [--keep]
//! ```
//!
//! The checkpoint goes to `/dev/shm/pie-cerebras-e2e/<sku>.zt` (removed unless
//! `--keep`). The device lock is held only while the shell is up.

use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_cerebras::DeviceBoot;
use engine_cerebras::serve::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Dtype, ParamSource, Platform, Request, Weight};

/// Prefill rows the synthetic fires take (`PIE_E2E_ROWS`, 8 by default).
/// Under a whole fire with kept pools the unified shape covers fires up
/// to this many rows; a larger fire is refused outright (pico on four PEs
/// cannot hold the budget's 24 rows as a whole fire, and the lane ladder
/// wants at least 18 token rows, so the probe stays below the budget).
fn probe_rows_of(max_tokens: u32) -> u32 {
    std::env::var("PIE_E2E_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .min(max_tokens)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let name = args.next().expect("usage: tiny_e2e <sku> [--keep]");
    let keep = args.any(|a| a == "--keep");
    let sku = dev_skus()
        .into_iter()
        .find(|s| s.name == name)
        .or_else(|| models::sku(&name))
        .or_else(|| models::skus().find(|s| s.name.starts_with(&name)))
        .unwrap_or_else(|| panic!("no SKU `{name}`"));
    let trace = (sku.trace)(Platform::Cerebras);
    let dir = PathBuf::from("/dev/shm/pie-cerebras-e2e");
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
        checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Cerebras)
            .unwrap_or_else(|why| panic!("the random checkpoint reads: {why}"));
        println!("{}: written and read", sku.name);
        return;
    }
    let source = ztensor_compat::index(&path).expect("the checkpoint indexes");
    let contract = checkpoint_dsl::own_contract(&source, &trace.params, 1, Platform::Cerebras)
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
    sku: &models::Sku,
    trace: model_dsl::Trace,
    contract: &checkpoint::contract::ModelContract,
    path: &Path,
) -> Result<String, String> {
    let context = 32;
    let budgets = engine::load::Budgets {
        max_lanes: 2,
        max_tokens: 24,
        page_size: 16,
        max_context: context,
        slots: 4,
        pages: 4,
        ..engine::load::Budgets::default()
    };
    let patches = engine_cerebras::api::patch_ladder(&trace, &budgets);
    let voxels = engine_cerebras::dit::voxel_ladder(&trace, Some(4096), Some(4), budgets.max_lanes);
    let t0 = Instant::now();
    let device_boot = DeviceBoot::default();
    let boot = |trace: model_dsl::Trace| Boot {
        trace,
        contract,
        checkpoint: path,
        budget: Budget::new(budgets.max_lanes, budgets.max_tokens),
        patches: patches.clone(),
        page_size: budgets.page_size,
        context,
        slots: budgets.slots,
        pages: budgets.pages,
        device: &device_boot,
    };
    // Whole-fire programs follow one model-wide shape: a dry shell traces
    // every class first and the real shell takes the union.
    let unified = if kernels_cerebras::program::whole_fire() {
        // `PIE_E2E_DRY=compile` also compiles what the dry shell traces.
        let compile = std::env::var("PIE_E2E_DRY").is_ok_and(|v| v == "compile");
        let mut dry = Shell::load_dry(boot(trace.clone()), voxels.clone(), compile)
            .map_err(|fault| format!("dry load: {fault}"))?;
        let probes = dry.synthetic_fires(sku.classify, probe_rows_of(budgets.max_tokens), true);
        if let Some(p) = probes.iter().find(|p| p.outcome.is_err()) {
            return Err(format!("dry class {} rows {}: {:?}", p.class, p.rows, p.outcome));
        }
        let mut texts = dry.device().dry_texts();
        // Fires that split into rows differently are traced once more
        // under one partition, so their rows unite.
        if let Some(partition) = Shell::unified_partition(&texts)
            && partition.row_phases.is_some()
        {
            let mut again = Shell::load_dry(boot(trace.clone()), voxels.clone(), compile)
                .map_err(|fault| format!("dry load: {fault}"))?;
            again.set_unified(Some(partition));
            let probes = again.synthetic_fires(sku.classify, probe_rows_of(budgets.max_tokens), true);
            if let Some(p) = probes.iter().find(|p| p.outcome.is_err()) {
                return Err(format!("dry class {} rows {} (partitioned): {:?}", p.class, p.rows, p.outcome));
            }
            texts = again.device().dry_texts();
        }
        let spec = Shell::unified_of_texts(&texts);
        eprintln!(
            "{}: unified {:?}",
            sku.name,
            spec.as_ref().map(|u| (
                u.row_phases.clone(),
                u.keep.len(),
                u.keep_chunks,
                u.rows.iter().map(|r| (r.kernels.len(), r.arena_words, r.rows, r.ops_words)).collect::<Vec<_>>()
            ))
        );
        // `PIE_E2E_DRY=1` stops here and sizes every program the dry shell
        // traced: its arena, op table and source lines (a program over a PE
        // fails to link, and the budget must say so first).
        if std::env::var_os("PIE_E2E_DRY").is_some() {
            // Under the unified shape every class should trace one program
            // text (one binary, one server).
            if spec.is_some() {
                let mut again = Shell::load_dry(boot(trace.clone()), voxels.clone(), compile)
                    .map_err(|fault| format!("dry load: {fault}"))?;
                again.set_unified(spec.clone());
                let probes = again.synthetic_fires(sku.classify, probe_rows_of(budgets.max_tokens), true);
                if let Some(p) = probes.iter().find(|p| p.outcome.is_err()) {
                    return Err(format!("dry class {} rows {} (unified): {:?}", p.class, p.rows, p.outcome));
                }
                let mut codes: Vec<(String, String)> = Vec::new();
                for text in again.device().dry_texts() {
                    for phase in text.split(engine_cerebras::trace::PHASE_SEPARATOR) {
                        if let Some(r) = engine_cerebras::trace::unrender(phase) {
                            codes.push((r.layout, r.pe));
                        }
                    }
                }
                let distinct = {
                    let mut d = codes.clone();
                    d.sort();
                    d.dedup();
                    d.len()
                };
                eprintln!("{}: unified: {} programs, {distinct} distinct program texts", sku.name, codes.len());
                if let Some(dir) = std::env::var_os("PIE_E2E_DRY_DUMP") {
                    let dir = std::path::PathBuf::from(dir);
                    let _ = std::fs::create_dir_all(&dir);
                    for (i, (layout, pe)) in codes.iter().enumerate() {
                        let _ = std::fs::write(dir.join(format!("{i}.layout.csl")), layout);
                        let _ = std::fs::write(dir.join(format!("{i}.pe.csl")), pe);
                    }
                }
            }
            for text in &texts {
                for phase in text.split(engine_cerebras::trace::PHASE_SEPARATOR) {
                    let Some(r) = engine_cerebras::trace::unrender(phase) else { continue };
                    let arena: Vec<&str> = r.pe.lines().filter(|l| l.starts_with("var arena") && l.contains(": [")).collect();
                    let ops = r.pe.lines().find(|l| l.starts_with("var ops:")).unwrap_or("");
                    let lines = r.pe.lines().filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("//")).count();
                    eprintln!("  program rect {:?} lines {lines} {:?} {ops}", r.manifest.rect, arena);
                }
            }
            return Ok(format!("{:<50} DRY", sku.name));
        }
        spec
    } else {
        None
    };
    let mut shell = Shell::load_with(boot(trace), voxels).map_err(|fault| format!("load: {fault}"))?;
    shell.set_unified(unified);
    let loaded = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let probes = shell.synthetic_fires(sku.classify, probe_rows_of(budgets.max_tokens), true);
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
fn agree(shell: &mut Shell, sku: &models::Sku) -> Result<String, String> {
    let word = |query_len: u32| (sku.classify)(&Request::new(query_len, false));
    let vocab = shell.out_width();
    let mut lcg = 0x2545_f491_4f6c_dd1du64;
    let prompt: Vec<u32> = (0..6)
        .map(|_| {
            lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((lcg >> 33) % u64::from(vocab.clamp(2, 1000))) as u32
        })
        .collect();
    let fault = |what: &'static str| move |f: engine_cerebras::Fault| format!("{what}: {f}");
    // `PIE_E2E_PROBE=1`: every token-rowed op output of the walk is read
    // back, and the first whose prefill row and walked row part is named.
    let probing = std::env::var_os("PIE_E2E_PROBE").is_some();
    if probing {
        let trace = shell.trace();
        let values: Vec<model_ir::ValueId> = trace
            .values
            .iter()
            .enumerate()
            .filter(|(_, d)| {
                matches!(d.def, model_ir::Def::Op(_))
                    && matches!(&d.ty, model_ir::Ty::Tensor { shape, .. } if shape.first() == Some(&model_ir::Dim::Tokens))
            })
            .map(|(at, _)| model_ir::ValueId(at as u32))
            .collect();
        shell.probe(values);
    }
    shell.open(0).map_err(fault("open"))?;
    let first_probes = if probing {
        let seated = [engine_cerebras::serve::Seated::of(Lane {
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
        let seated = [engine_cerebras::serve::Seated::of(Lane {
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
                model_ir::Def::Op(n) => n as usize,
                _ => 0,
            };
            let op = model_ir::Operands::name(&shell.trace().nodes[node].op);
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
fn write_random(trace: &model_dsl::Trace, path: &Path) -> u64 {
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
    let mut params: Vec<&model_dsl::Param> = trace.params.iter().collect();
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
                            f16_bits(v).to_le_bytes()
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
                    .flat_map(|_| f16_bits(next()).to_le_bytes())
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

/// `v` rounded to IEEE binary16 (nearest even), as its bits.
fn f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | u16::from(mant != 0) << 9;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (mant | 0x80_0000) >> (1 - e);
        let round = (m >> 13)
            + u32::from((m & 0x1fff) > 0x1000 || ((m & 0x1fff) == 0x1000 && (m >> 13) & 1 == 1));
        return sign | round as u16;
    }
    let mut half = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    sign | half as u16
}

/// Development models this backend brings up on: the tiny Qwen with dense
/// bf16 weights, and a pico Qwen that fits one PE. Neither is a catalog SKU.
fn dev_skus() -> Vec<&'static models::Sku> {
    let mut out = Vec::new();
    for (name, trace) in [
        (
            "qwen35-tiny-bf16",
            (|p: Platform| {
                model_dsl::trace_hybrid(
                    "qwen35-tiny-bf16",
                    &models::qwen_3::model::Model::tiny(Dtype::Bf16, Dtype::Bf16, 1),
                    p,
                )
            }) as model_dsl::TraceFn,
        ),
        (
            "qwen35-pico",
            (|p: Platform| {
                model_dsl::trace_hybrid(
                    "qwen35-pico",
                    &models::qwen_3::model::Model::pico(Dtype::Bf16, Dtype::Bf16, 1),
                    p,
                )
            }) as model_dsl::TraceFn,
        ),
    ] {
        let Some(base) = models::skus().find(|s| s.name.starts_with("qwen35-tiny")) else {
            continue;
        };
        out.push(&*Box::leak(Box::new(models::Sku {
            name: name.to_string(),
            recipe: base.recipe,
            trace,
            classify: base.classify,
            import: base.import,
            template: base.template,
            tokenizer: base.tokenizer,
            diffusion: base.diffusion,
            generative: base.generative.clone(),
        })));
    }
    out
}
