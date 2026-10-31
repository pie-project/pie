//! The head's eta VM driven by the engine's host side ([`guest4::wafer`]) on
//! the fabric simulator, on a head-only image (head2d with the VM on its VM
//! columns): `vm_sim <compiled program dir> <ckpt dir> <VM case json>
//! <steps> <out dir>`. The head computes logits from a fixed hidden state
//! over the checkpoint's tied embedding; the VM samples every step with the
//! case's table and seeds. Writes `<out dir>/plane.f32` (the logits row the
//! VM received, x 1/rms, vocabulary order) and `<out dir>/vm.json` (the out
//! ring's rows and the head's token), for holding the VM's samples against
//! eta's interpreter on that row.

mod common;

use std::path::Path;

use engine_cerebras::sdk::{DataType, Order, Runtime, Sdk};
use engine_cerebras::spatial::ckpt::Ckpt;
use engine_cerebras::spatial::guest4::lane;
use engine_cerebras::spatial::guest4::wafer::{self, VmLayout};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let (program, ckpt, case, steps, out) = (
        Path::new(&args[1]),
        Path::new(&args[2]),
        &args[3],
        args[4].parse::<usize>().expect("steps"),
        Path::new(&args[5]),
    );
    let json = common::read_json(program.join("out.json"))?;
    let get = |k: &str| {
        json["params"][k]
            .as_str()
            .and_then(|v| v.parse::<usize>().ok())
    };
    let p = |k: &str| get(k).ok_or(format!("no param {k}"));
    let lay = VmLayout::of(get).ok_or("not a head VM image")?;
    let (w, hd, c, hc, vr, vocab) = (p("W")?, p("H")?, p("C")?, p("Hc")?, p("Vr")?, p("vocab")?);
    if p("WF32")? != 1 || p("UNTIED")? != 0 {
        return Err("this check places f32 tied heads".into());
    }
    let (nb, f, hid, r2x) = (lay.blocks, lay.feeder, c * hc, lay.root);
    let cb = lay.block_cols;

    // The case: its table, homes, outputs and seeds. The head image
    // takes no per-step input and keeps no lane banks: those are left out.
    let case = common::read_json(case)?;
    let (mut table, seeds) = wafer::case_table(&case)?;
    table.per_step.clear();
    table.bank_regs.clear();
    table.bank_pool.clear();

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
    let platform = common::simulator(5);
    let mut rt = Runtime::new(sdk, program, platform, "WARNING").map_err(|e| e.to_string())?;
    let e = |e: engine_cerebras::sdk::Error| e.to_string();
    rt.load();
    rt.run();
    rt.launch("init", &[]).map_err(e)?;
    let b32 = DataType::Bits32;
    for &x in &head_cols {
        for (y0, rows) in [(0, 0..f), (f + 1, f..hd)] {
            let words: Vec<u32> = rows.clone().flat_map(|ry| tile(x, ry)).collect();
            rt.memcpy_h2d(
                "wt",
                &words,
                common::rect(x, y0, 1, rows.len()),
                n,
                Order::RowMajor,
                b32,
            )
            .map_err(e)?;
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
    rt.memcpy_h2d(
        "xin",
        &feed(&x),
        common::rect(0, f, w, 1),
        hc,
        Order::RowMajor,
        b32,
    )
    .map_err(e)?;
    rt.memcpy_h2d(
        "gam",
        &feed(&norm.data[..hid]),
        common::rect(0, f, w, 1),
        hc,
        Order::RowMajor,
        b32,
    )
    .map_err(e)?;

    // The VM: the engine's own load, ring and readout.
    // `gring`'s words, as the head image's PE ELFs define it.
    let words = lane::symbol_words(program, "gring").ok_or("no PE image defines `gring`")?;
    let ring = lane::read_ring(&mut rt, lay.hub(), words)?;
    if let Some(why) = ring.refuses_outputs(&table.outputs) {
        return Err(why);
    }
    wafer::load(&mut rt, &lay, &table, &seeds)?;
    rt.launch("compute", &[steps as u32]).map_err(e)?;
    let rows = wafer::read_out(&mut rt, &lay, ring, 0, steps)?;
    let mut tok = [0u32; 4];
    let bm = (nb / p("Gb")? / 2) * p("Gb")? + p("Gb")? / 2;
    rt.memcpy_d2h(
        "tok",
        &mut tok,
        common::rect(bm * cb + r2x, f, 1, 1),
        4,
        Order::RowMajor,
        b32,
    )
    .map_err(e)?;
    // What the VM sampled from: each VM PE's received row and 1/rms.
    let mut plane = Vec::with_capacity(nb * hd * vr);
    for b in 0..nb {
        let mut vin = vec![0u32; (hd + 1) * (vr + 1)];
        rt.memcpy_d2h(
            "vin",
            &mut vin,
            common::rect(lay.column(b), 0, 1, hd + 1),
            vr + 1,
            Order::RowMajor,
            b32,
        )
        .map_err(e)?;
        for pe in (0..=hd).filter(|&pe| pe != f) {
            let r = &vin[pe * (vr + 1)..(pe + 1) * (vr + 1)];
            let ys = f32::from_bits(r[vr]);
            plane.extend(r[..vr].iter().map(|&v| f32::from_bits(v) * ys));
        }
    }
    rt.stop();
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let bytes: Vec<u8> = plane[..vocab]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(out.join("plane.f32"), bytes).map_err(|e| e.to_string())?;
    let report = serde_json::json!({
        "ring": rows, "ring_depth": ring.depth, "ring_width": ring.width, "steps": steps,
        "head_token": f32::from_bits(tok[1]) as u32, "case": case["name"],
    });
    std::fs::write(out.join("vm.json"), report.to_string()).map_err(|e| e.to_string())?;
    println!("{report}");
    Ok(())
}
