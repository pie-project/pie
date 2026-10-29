//! The head VM (`payload/guest4h3`: head2d with the eta VM on its VM
//! columns) driven by the engine's host side ([`guest4::wafer`]) on the
//! fabric simulator: `vm_sim <compiled program dir> <ckpt dir> <etagen case
//! json> <steps> <out dir>`. The head computes logits from a fixed hidden
//! state over the checkpoint's tied embedding; the VM samples every step
//! with the case's table and seeds. Writes `<out dir>/plane.f32` (the logits
//! row the VM received, x 1/rms, vocabulary order) and `<out dir>/vm.json`
//! (the out ring's rows and the head's token), which `check_dump.py` holds
//! against eta's interpreter on that row.

use std::path::Path;

use engine_cerebras::sdk::{DataType, Order, Platform, Rect, Runtime, Sdk, Simulator, Target};
use engine_cerebras::spatial::ckpt::Ckpt;
use engine_cerebras::spatial::guest4::wafer::{self, Seed, VmLayout};
use engine_cerebras::spatial::guest4::{Home, Table, lane};

fn rect(x: usize, y: usize, w: usize, h: usize) -> Rect {
    Rect {
        x: x as i32,
        y: y as i32,
        w: w as i32,
        h: h as i32,
    }
}

fn dtype(s: &str) -> eta_ir::Dtype {
    match s {
        "F32" => eta_ir::Dtype::F32,
        "I32" => eta_ir::Dtype::I32,
        "Bool" => eta_ir::Dtype::Bool,
        _ => eta_ir::Dtype::U32,
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let (program, ckpt, case, steps, out) =
        (Path::new(&args[1]), Path::new(&args[2]), &args[3], args[4].parse::<usize>().expect("steps"), Path::new(&args[5]));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(program.join("out.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let get = |k: &str| json["params"][k].as_str().and_then(|v| v.parse::<usize>().ok());
    let p = |k: &str| get(k).ok_or(format!("no param {k}"));
    let lay = VmLayout::of(get).ok_or("not a head VM image")?;
    let (w, hd, c, hc, vr, vocab) = (p("W")?, p("H")?, p("C")?, p("Hc")?, p("Vr")?, p("vocab")?);
    if p("WF32")? != 1 || p("UNTIED")? != 0 {
        return Err("this check places f32 tied heads".into());
    }
    let (nb, f, hid, r2x) = (lay.blocks, lay.feeder, c * hc, lay.root);
    let cb = lay.block_cols;

    // The case: etagen's table, homes, outputs and seeds.
    let case: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(case).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let u = |v: &serde_json::Value| v.as_u64().unwrap_or(0) as u32;
    // Instructions as six-word arrays (or one flat list of them).
    let flat: Vec<u32> = case["table"]
        .as_array()
        .ok_or("no table")?
        .iter()
        .flat_map(|i| match i.as_array() {
            Some(words) => words.iter().map(u).collect(),
            None => vec![u(i)],
        })
        .collect();
    let ins: Vec<[u32; 6]> = flat.chunks_exact(6).map(|w| [w[0], w[1], w[2], w[3], w[4], w[5]]).collect();
    let homes: Vec<Home> = case["homes"]
        .as_array()
        .ok_or("no homes")?
        .iter()
        .map(|h| Home {
            chan: u(&h["chan"]),
            kind: if h["kind"] == "pool" { "pool" } else { "wide" },
            at: u(&h["at"]),
            n: u(&h["n"]),
            dtype: dtype(h["dtype"].as_str().unwrap_or("U32")),
        })
        .collect();
    let table = Table {
        ins,
        pre_len: case["pre_len"].as_u64().unwrap_or(0) as usize,
        homes,
        outputs: case["outputs"]
            .as_array()
            .ok_or("no outputs")?
            .iter()
            .map(|o| (u(&o["chan"]), u(&o["slot"]), u(&o["n"]), dtype(o["dtype"].as_str().unwrap_or("U32"))))
            .collect(),
        wide_outputs: Vec::new(),
        per_step: Vec::new(),
        tok_home: case["tok_home"].as_u64().map(|t| t as u32),
        nreg: u(&case["nreg"]),
        pool_words: u(&case["pool_words"]),
        feat: u(&case["feat"]),
        remat: false,
        bank_regs: Vec::new(),
        bank_pool: Vec::new(),
    };
    let seeds: Vec<Seed> = table
        .homes
        .iter()
        .filter_map(|h| {
            let words = case["seeds"][h.chan.to_string()].as_array()?;
            Some(Seed::of(h, words.iter().map(u).collect()))
        })
        .collect();

    // The head's weights: the tied embedding's first `hid` dims over the
    // region (row (b·H + ry)·Vr + j on PE (b·C + c, ry), dims c·Hc.., dim-major,
    // one leading pad word), the final norm, a fixed hidden state.
    let ck = Ckpt::open(ckpt)?;
    let emb = ck.get("model.embed_tokens.weight")?;
    let norm = ck.get("model.norm.weight")?;
    let cols = emb.shape[1];
    let n = hc * vr + 1;
    let head_cols: Vec<usize> = (0..w).filter(|x| x % cb != r2x + 1).collect();
    let tile = |x: usize, ry: usize| -> Vec<u32> {
        let (b, cc) = (x / cb, x % cb);
        let cc = if cc > r2x { cc - 1 } else { cc };
        let mut t = vec![0u32; n];
        for h in 0..hc {
            for j in 0..vr {
                let row = (b * hd + ry) * vr + j;
                if row < vocab {
                    t[1 + h * vr + j] = emb.data[row * cols + cc * hc + h].to_bits();
                }
            }
        }
        t
    };
    let sdk = Sdk::open().map_err(|e| e.to_string())?;
    let platform = Platform::Simulator(Simulator {
        num_threads: std::env::var("PIE_SIM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(5),
        ..Simulator::new(Target::Wse3)
    });
    let mut rt = Runtime::new(sdk, program, platform, "WARNING").map_err(|e| e.to_string())?;
    let e = |e: engine_cerebras::sdk::Error| e.to_string();
    rt.load();
    rt.run();
    rt.launch("init", &[]).map_err(e)?;
    let b32 = DataType::Bits32;
    for &x in &head_cols {
        for (y0, rows) in [(0, 0..f), (f + 1, f..hd)] {
            let words: Vec<u32> = rows.clone().flat_map(|ry| tile(x, ry)).collect();
            rt.memcpy_h2d("wt", &words, rect(x, y0, 1, rows.len()), n, Order::RowMajor, b32).map_err(e)?;
        }
    }
    let x: Vec<f32> = (0..hid).map(|i| ((i as f32) * 0.37).sin() * 2.0).collect();
    let feed = |v: &[f32]| -> Vec<u32> {
        (0..w)
            .flat_map(|xc| {
                let (cc, vm) = (xc % cb, xc % cb == r2x + 1);
                let cc = if cc > r2x { cc - 1 } else { cc };
                (0..hc).map(move |h| if vm { 0 } else { v[cc * hc + h].to_bits() })
            })
            .collect()
    };
    rt.memcpy_h2d("xin", &feed(&x), rect(0, f, w, 1), hc, Order::RowMajor, b32).map_err(e)?;
    rt.memcpy_h2d("gam", &feed(&norm.data[..hid]), rect(0, f, w, 1), hc, Order::RowMajor, b32).map_err(e)?;

    // The VM: the engine's own load, ring and readout.
    let ring = lane::read_ring(&mut rt, lay.hub(), lane::ring_words(program)?)?;
    if let Some(why) = ring.refuses_outputs(&table.outputs) {
        return Err(why);
    }
    wafer::load(&mut rt, &lay, &table, &seeds)?;
    rt.launch("compute", &[steps as u32]).map_err(e)?;
    let rows = wafer::read_out(&mut rt, &lay, ring, 0, steps)?;
    let mut tok = [0u32; 4];
    let bm = (nb / p("Gb")? / 2) * p("Gb")? + p("Gb")? / 2;
    rt.memcpy_d2h("tok", &mut tok, rect(bm * cb + r2x, f, 1, 1), 4, Order::RowMajor, b32).map_err(e)?;
    // What the VM sampled from: each VM PE's received row and 1/rms.
    let mut plane = Vec::with_capacity(nb * hd * vr);
    for b in 0..nb {
        let mut vin = vec![0u32; (hd + 1) * (vr + 1)];
        rt.memcpy_d2h("vin", &mut vin, rect(lay.column(b), 0, 1, hd + 1), vr + 1, Order::RowMajor, b32)
            .map_err(e)?;
        for pe in (0..=hd).filter(|&pe| pe != f) {
            let r = &vin[pe * (vr + 1)..(pe + 1) * (vr + 1)];
            let ys = f32::from_bits(r[vr]);
            plane.extend(r[..vr].iter().map(|&v| f32::from_bits(v) * ys));
        }
    }
    rt.stop();
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let bytes: Vec<u8> = plane[..vocab].iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out.join("plane.f32"), bytes).map_err(|e| e.to_string())?;
    let report = serde_json::json!({
        "ring": rows, "ring_depth": ring.depth, "ring_width": ring.width, "steps": steps,
        "head_token": f32::from_bits(tok[1]) as u32, "case": case["name"],
    });
    std::fs::write(out.join("vm.json"), report.to_string()).map_err(|e| e.to_string())?;
    println!("{report}");
    Ok(())
}
