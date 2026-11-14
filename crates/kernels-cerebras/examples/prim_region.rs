//! A root region's standalone test (tools/wafer/prims/reg): the ops of a run of flux2-mini's
//! denoise nodes lowered to a region (poem-compiler `fuse::chain::lower_run`), printed for a root
//! (`emit::region`, P4), with its inputs and oracles from the step's captured values (engine-cpu's
//! `flux_2_minis_primitive_goldens`):
//!
//! `cargo run --release -p kernels-cerebras --example prim_region -- <goldens> <out> <nodes,..>
//!  <rows from> <rows> <lane> <chunk> [round16]`
//!
//! Writes `<out>/reg.csl` (the printed module), `meta.json`, `rows.npy` (each row's record: the
//! row params' chunks, then its scalars, root after root: `[rows, roots, NROW C + NSC]`),
//! `res.npy` (`[roots, NRES C]`), `tabs.npy` (`[roots, NTAB]`), `want.npy` (kernels_ir::interp's
//! results, `[rows, results, width]`) and `trace.npy` (the step's own values of the results).
//! Prints the region's ops and interp's distance from the trace.

use std::collections::BTreeMap;
use std::path::Path;

use kernels_cerebras::emit::region::{self, ParamAt, Place};
use kernels_ir::Shape;
use kernels_ir::interp::{self, Arg};
use poem_ir::Platform;

const MINI: &str = "flux2-mini-bf16-kv-bf16";

fn read_npy(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let h = std::str::from_utf8(&b[10..10 + hl]).expect("an npy header");
    let shape: Vec<usize> = h[h.find('(').unwrap() + 1..h.find(')').unwrap()]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let body = &b[10 + hl..];
    let ints = h.contains("<i4");
    let v = body
        .chunks(4)
        .map(|c| {
            let w = [c[0], c[1], c[2], c[3]];
            if ints {
                i32::from_le_bytes(w) as f32
            } else {
                f32::from_le_bytes(w)
            }
        })
        .collect();
    (shape, v)
}

fn write_npy(path: &Path, shape: &[usize], v: &[f32]) {
    let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
    let tail = if shape.len() == 1 { "," } else { "" };
    let mut h = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}{tail}), }}",
        dims.join(", ")
    );
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, out).expect("an npy file");
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (gold, out) = (Path::new(&a[1]), Path::new(&a[2]));
    let run: Vec<u32> = a[3]
        .split(',')
        .map(|n| n.parse().expect("a node"))
        .collect();
    let (from, rows, lane, chunk) = (
        a[4].parse::<usize>().unwrap(),
        a[5].parse::<usize>().unwrap(),
        a[6].parse::<usize>().unwrap(),
        a[7].parse::<u32>().unwrap(),
    );
    let round16 = a.get(8).is_some_and(|s| s == "1");
    let trace = poem_compiler::catalog::repository()
        .parse(MINI)
        .expect("the catalog holds flux2-mini")
        .trace(Platform::Cerebras);
    let chain = poem_compiler::fuse::chain::lower_run(&trace, &run).expect("the run lowers");
    let reg = &chain.region;
    let width = reg.width as usize;
    eprintln!(
        "region: width {width}, {} params, {} ops, results {:?}, rounds {}",
        reg.params.len(),
        reg.ops.len(),
        reg.results,
        reg.rounds.len()
    );
    // every captured value by id
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(gold.join("nodes.json")).unwrap()).unwrap();
    let mut file_of: BTreeMap<u64, String> = BTreeMap::new();
    for node in index.as_object().unwrap().values() {
        for (f, v) in node["values"].as_object().unwrap() {
            file_of.insert(v.as_u64().unwrap(), f.clone());
        }
    }
    let load = |v: poem_ir::ValueId| {
        let f = file_of
            .get(&u64::from(v.0))
            .unwrap_or_else(|| panic!("value {} is not captured", v.0));
        read_npy(&gold.join(f))
    };
    // each parameter's place and its value at row r
    let mut place = Vec::new();
    let mut data = Vec::new();
    for (input, &shape) in chain.ins.iter().zip(&reg.params) {
        let (s, v) = load(input.value);
        // a weight (shared by every row) is its elements whole
        let w = if input.shared {
            v.len()
        } else {
            *s.last().unwrap()
        };
        let at = if shape == Shape::Scalar {
            ParamAt::Scalar
        } else if input.by.is_some() || input.shared || s.len() == 1 {
            ParamAt::Resident
        } else {
            ParamAt::Row
        };
        place.push(at);
        data.push((
            input.by.is_some(),
            input.from as usize,
            w,
            input.shared || s.len() == 1,
            v,
        ));
    }
    let arg = |k: usize, r: usize| -> Arg {
        let (by, from, w, flat, v) = &data[k];
        let row = if *flat {
            0
        } else if *by {
            lane
        } else {
            r
        };
        match place[k] {
            ParamAt::Scalar => Arg::Scalar(v[row * w + from]),
            _ => Arg::Row(
                (0..width)
                    .map(|e| {
                        if *flat {
                            v[(from + e) % w]
                        } else {
                            v[row * w + from + e]
                        }
                    })
                    .collect(),
            ),
        }
    };
    let printed = region::print(
        reg,
        &Place {
            chunk,
            roots: reg.width / chunk,
            params: place.clone(),
            round16,
            rope_lv: 4,
            rope_bits: 3,
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));
    std::fs::create_dir_all(out).unwrap();
    std::fs::write(out.join("reg.csl"), &printed.csl).unwrap();
    let roots = reg.width / chunk;
    let c = chunk as usize;
    let nrow = place.iter().filter(|p| **p == ParamAt::Row).count();
    let nsc = place.iter().filter(|p| **p == ParamAt::Scalar).count();
    let nres = place.iter().filter(|p| **p == ParamAt::Resident).count();
    let rec = nrow * c + nsc;
    let mut rowv = vec![0f32; rows * roots as usize * rec.max(1)];
    let mut want = vec![0f32; rows * reg.results.len() * width];
    let mut tracev = vec![0f32; rows * reg.results.len() * width];
    let outs: Vec<(Vec<usize>, Vec<f32>)> = chain.outs.iter().map(|o| load(o.value)).collect();
    let (mut d2, mut n2, mut dt2) = (0f64, 0f64, 0f64);
    for t in 0..rows {
        let r = from + t;
        let args: Vec<Arg> = (0..reg.params.len()).map(|k| arg(k, r)).collect();
        let res = interp::eval(reg, &args).expect("interp runs the region");
        for root in 0..roots as usize {
            let base = (t * roots as usize + root) * rec.max(1);
            let (mut i, mut j) = (0, 0);
            for (k, p) in place.iter().enumerate() {
                match (p, &args[k]) {
                    (ParamAt::Row, Arg::Row(v)) => {
                        rowv[base + i * c..base + (i + 1) * c]
                            .copy_from_slice(&v[root * c..(root + 1) * c]);
                        i += 1;
                    }
                    (ParamAt::Scalar, Arg::Scalar(x)) => {
                        rowv[base + nrow * c + j] = *x;
                        j += 1;
                    }
                    _ => {}
                }
            }
        }
        for (k, row) in res.iter().enumerate() {
            let (s, v) = &outs[k];
            let w = *s.last().unwrap();
            for e in 0..width {
                want[(t * reg.results.len() + k) * width + e] = row[e];
                let tv = v[r * w + e];
                tracev[(t * reg.results.len() + k) * width + e] = tv;
                d2 += f64::from(row[e] - tv).powi(2);
                n2 += f64::from(tv).powi(2);
                dt2 += 1.0;
            }
        }
    }
    let mut resv = vec![0f32; roots as usize * (nres * c).max(1)];
    for root in 0..roots as usize {
        let mut i = 0;
        for (k, p) in place.iter().enumerate() {
            if *p == ParamAt::Resident {
                if let Arg::Row(v) = arg(k, from) {
                    let o = root * (nres * c).max(1) + i * c;
                    resv[o..o + c].copy_from_slice(&v[root * c..(root + 1) * c]);
                }
                i += 1;
            }
        }
    }
    let ntab: usize = printed
        .tables
        .iter()
        .map(|t| t.of_chunk.iter().map(Vec::len).max().unwrap_or(0))
        .sum();
    let mut tabs = vec![0f32; roots as usize * ntab.max(1)];
    for root in 0..roots as usize {
        let mut off = 0;
        for t in &printed.tables {
            let n = t.of_chunk.iter().map(Vec::len).max().unwrap_or(0);
            let v = &t.of_chunk[root];
            tabs[root * ntab.max(1) + off..root * ntab.max(1) + off + v.len()].copy_from_slice(v);
            off += n;
        }
    }
    let nout = reg.results.len();
    write_npy(
        &out.join("rows.npy"),
        &[rows, roots as usize, rec.max(1)],
        &rowv,
    );
    write_npy(
        &out.join("res.npy"),
        &[roots as usize, (nres * c).max(1)],
        &resv,
    );
    write_npy(&out.join("tabs.npy"), &[roots as usize, ntab.max(1)], &tabs);
    write_npy(&out.join("want.npy"), &[rows, nout, width], &want);
    write_npy(&out.join("trace.npy"), &[rows, nout, width], &tracev);
    let meta = serde_json::json!({
        "nodes": run, "width": width, "chunk": chunk, "roots": roots, "rows": rows, "rec": rec.max(1),
        "nres": (nres * c).max(1), "ntab": ntab.max(1), "nout": nout, "all_reduces": printed.all_reduces,
        "params": printed.params.iter().map(|(n, p)| format!("{n}:{p:?}")).collect::<Vec<_>>(),
        "tables": printed.tables.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
        "ops": reg.ops.len(), "round16": round16,
    });
    std::fs::write(
        out.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
    println!(
        "PRIM_REGION_HOST nodes {run:?}: interp vs trace rel L2 {:.3e} over {} values; {} all-reduces a row",
        (d2 / n2).sqrt(),
        dt2,
        printed.all_reduces
    );
}
