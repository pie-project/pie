//! The wafer program host's whole output for one artifact and prep, on a recording device: the
//! artifact `split-geo build` wrote, read as the engine reads it, then what `SplitModel::open`
//! places (every PE's words of every symbol, rcfg in full),
//! the calls it makes, the values it derives, and one greedy launch's host
//! traffic. Two runs (before and after a change) compare file for file
//! (`crates/kernels-cerebras/tools/split/baseline`).
//!
//! `split_baseline <artifact dir> <prep dir> <prefix> <vocab> <out dir>`
//!
//! Writes into `<out dir>`: `pe.tsv` (`sym x y n bits sha256` a PE, the last
//! write of each symbol, sorted), `rcfg.txt` (`x y w..` a PE), `calls.log`
//! (every device call in order, copies as `sym x y w h n bits sha256`) and
//! `derived.json` (the model's public fields after open).

use std::collections::BTreeMap;
use std::io::Write;

use engine_cerebras::sdk::{self, DataType, Order, Rect};
use engine_cerebras::spatial::device::Device;
use engine_cerebras::spatial::split::{SplitModel, SplitSpec};
use kernels_cerebras::spatial::artifact::Artifact;
use sha2::{Digest, Sha256};

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha(words: &[u32]) -> String {
    let mut h = Sha256::new();
    for w in words {
        h.update(w.to_le_bytes());
    }
    hex(&h.finalize())
}

#[derive(Default)]
struct Rec {
    /// (sym, y, x) -> (n, bits, sha256 of the PE's words)
    pe: BTreeMap<(String, u32, u32), (usize, u32, String)>,
    /// (y, x) -> the PE's last rcfg words
    rcfg: BTreeMap<(u32, u32), Vec<u32>>,
    calls: Vec<String>,
    /// `SPLIT_BASELINE_LOGITS=1`: reads of `x` / `lgt` return fixed non-zero values (the logits
    /// step's host math is then compared too).
    fill: bool,
}

impl Rec {
    fn h2d(&mut self, s: &str, d: &[u32], r: Rect, n: usize, o: Order, dt: DataType) {
        let (w, h) = (r.w as usize, r.h as usize);
        assert_eq!(d.len(), w * h * n, "{s}");
        let bits = if dt == DataType::Bits16 { 16 } else { 32 };
        self.calls.push(format!(
            "h2d {s} {} {} {} {} {n} {bits} {:?} {}",
            r.x,
            r.y,
            r.w,
            r.h,
            o as u8,
            sha(d)
        ));
        for j in 0..h {
            for i in 0..w {
                // the PE's n words (row-major: rows, then columns, then the PE's elements)
                let at = |k: usize| match o {
                    Order::RowMajor => d[(j * w + i) * n + k],
                    Order::ColMajor => d[(i * h + j) * n + k],
                };
                let v: Vec<u32> = (0..n).map(at).collect();
                let (x, y) = (r.x as u32 + i as u32, r.y as u32 + j as u32);
                if s == "rcfg" {
                    self.rcfg.insert((y, x), v.clone());
                }
                self.pe.insert((s.to_string(), y, x), (n, bits, sha(&v)));
            }
        }
    }
}

impl Device for Rec {
    fn memcpy_h2d(
        &mut self,
        s: &str,
        d: &[u32],
        r: Rect,
        n: usize,
        o: Order,
        dt: DataType,
    ) -> sdk::Result<()> {
        self.h2d(s, d, r, n, o, dt);
        Ok(())
    }
    fn memcpy_d2h(
        &mut self,
        s: &str,
        out: &mut [u32],
        r: Rect,
        n: usize,
        _: Order,
        dt: DataType,
    ) -> sdk::Result<()> {
        let bits = if dt == DataType::Bits16 { 16 } else { 32 };
        self.calls.push(format!(
            "d2h {s} {} {} {} {} {n} {bits}",
            r.x, r.y, r.w, r.h
        ));
        out.fill(0);
        if self.fill && (s == "x" || s == "lgt") {
            for (i, v) in out.iter_mut().enumerate() {
                let k = (i * 7 + r.x as usize * 3 + r.y as usize) % 13;
                *v = (0.25 + k as f32 / 16.0).to_bits();
            }
        }
        Ok(())
    }
    fn memcpy_h2d_rowbcast(
        &mut self,
        s: &str,
        d: &[u32],
        r: Rect,
        n: usize,
        dt: DataType,
    ) -> sdk::Result<()> {
        self.calls.push(format!(
            "rowbcast {s} {} {} {} {} {n} {:?} {}",
            r.x,
            r.y,
            r.w,
            r.h,
            dt as u8,
            sha(d)
        ));
        Ok(())
    }
    fn memcpy_h2d_nonblock(
        &mut self,
        s: &str,
        d: Vec<u32>,
        r: Rect,
        n: usize,
        o: Order,
        dt: DataType,
    ) -> sdk::Result<()> {
        self.h2d(s, &d, r, n, o, dt);
        Ok(())
    }
    fn launch(&mut self, name: &str, args: &[u32]) -> sdk::Result<()> {
        self.calls.push(format!("launch {name} {args:?}"));
        Ok(())
    }
    fn launch_nonblock(&mut self, name: &str, args: &[u32]) -> sdk::Result<()> {
        self.calls.push(format!("launch_nb {name} {args:?}"));
        Ok(())
    }
    fn wait_pending(&mut self) {
        self.calls.push("wait".into());
    }
    fn has_symbol(&self, _: &str) -> bool {
        true
    }
}

/// The recorder behind the model's `Box<dyn Device>`, kept reachable after open.
struct Shared(std::rc::Rc<std::cell::RefCell<Rec>>);

impl Device for Shared {
    fn memcpy_h2d(
        &mut self,
        s: &str,
        d: &[u32],
        r: Rect,
        n: usize,
        o: Order,
        dt: DataType,
    ) -> sdk::Result<()> {
        self.0.borrow_mut().memcpy_h2d(s, d, r, n, o, dt)
    }
    fn memcpy_d2h(
        &mut self,
        s: &str,
        out: &mut [u32],
        r: Rect,
        n: usize,
        o: Order,
        dt: DataType,
    ) -> sdk::Result<()> {
        self.0.borrow_mut().memcpy_d2h(s, out, r, n, o, dt)
    }
    fn memcpy_h2d_rowbcast(
        &mut self,
        s: &str,
        d: &[u32],
        r: Rect,
        n: usize,
        dt: DataType,
    ) -> sdk::Result<()> {
        self.0.borrow_mut().memcpy_h2d_rowbcast(s, d, r, n, dt)
    }
    fn memcpy_h2d_nonblock(
        &mut self,
        s: &str,
        d: Vec<u32>,
        r: Rect,
        n: usize,
        o: Order,
        dt: DataType,
    ) -> sdk::Result<()> {
        self.0.borrow_mut().memcpy_h2d_nonblock(s, d, r, n, o, dt)
    }
    fn launch(&mut self, name: &str, args: &[u32]) -> sdk::Result<()> {
        self.0.borrow_mut().launch(name, args)
    }
    fn launch_nonblock(&mut self, name: &str, args: &[u32]) -> sdk::Result<()> {
        self.0.borrow_mut().launch_nonblock(name, args)
    }
    fn wait_pending(&mut self) {
        self.0.borrow_mut().wait_pending()
    }
    fn has_symbol(&self, _: &str) -> bool {
        true
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 6 {
        eprintln!("split_baseline <artifact dir> <prep dir> <prefix> <vocab> <out dir>");
        std::process::exit(2);
    }
    let spec = SplitSpec::parse(&format!("{}:{}", a[2], a[3])).unwrap();
    let vocab: usize = a[4].parse().unwrap();
    let out = std::path::Path::new(&a[5]);
    std::fs::create_dir_all(out).unwrap();
    let logits = std::env::var("SPLIT_BASELINE_LOGITS").is_ok_and(|v| v == "1");
    let rec = std::rc::Rc::new(std::cell::RefCell::new(Rec {
        fill: logits,
        ..Rec::default()
    }));
    let t0 = std::time::Instant::now();
    let mut derived = serde_json::Map::new();
    let opened = Artifact::read(std::path::Path::new(&a[1]))
        .and_then(|art| SplitModel::open(Box::new(Shared(rec.clone())), &spec, &art, vocab));
    match opened {
        Ok(mut m) => {
            derived.insert("vocab".into(), m.vocab.into());
            derived.insert("hidden".into(), m.hidden.into());
            derived.insert("cap".into(), m.cap.into());
            derived.insert("positions".into(), m.positions.into());
            rec.borrow_mut()
                .calls
                .push("-- greedy [1, 2, 3] start 0 extra 3".into());
            // (a zero device: the ring reads 0; the host's writes and launches are what count)
            let g = m.greedy(&[1, 2, 3], 0, 3);
            derived.insert("greedy".into(), format!("{g:?}").into());
            derived.insert("launches".into(), m.launches.into());
            derived.insert("steps".into(), m.steps.into());
            derived.insert("fed".into(), format!("{:?}", m.fed).into());
            if logits {
                rec.borrow_mut()
                    .calls
                    .push("-- forward [5, 6] logits".into());
                let start = m.fed.len();
                match m.forward(&[5, 6], start, true) {
                    Ok((am, l)) => {
                        let l = l.unwrap_or_default();
                        let bits: Vec<u32> = l.iter().map(|v| v.to_bits()).collect();
                        derived.insert("logits_argmax".into(), format!("{am:?}").into());
                        derived.insert("logits_sha".into(), sha(&bits).into());
                        derived.insert(
                            "logits_head".into(),
                            format!("{:?}", &l[..l.len().min(4)]).into(),
                        );
                    }
                    Err(e) => {
                        derived.insert("logits_error".into(), e.into());
                    }
                }
            }
        }
        Err(e) => {
            derived.insert("open_error".into(), e.into());
        }
    }
    eprintln!("opened {:.1} s", t0.elapsed().as_secs_f64());
    let r = rec.borrow();
    let mut f = std::io::BufWriter::new(std::fs::File::create(out.join("pe.tsv")).unwrap());
    for ((s, y, x), (n, bits, h)) in &r.pe {
        writeln!(f, "{s}\t{x}\t{y}\t{n}\t{bits}\t{h}").unwrap();
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(out.join("rcfg.txt")).unwrap());
    for ((y, x), w) in &r.rcfg {
        let ws: Vec<String> = w.iter().map(|v| v.to_string()).collect();
        writeln!(f, "{x} {y} {}", ws.join(" ")).unwrap();
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(out.join("calls.log")).unwrap());
    for c in &r.calls {
        writeln!(f, "{c}").unwrap();
    }
    std::fs::write(
        out.join("derived.json"),
        serde_json::to_string_pretty(&serde_json::Value::Object(derived)).unwrap() + "\n",
    )
    .unwrap();
    eprintln!("written {:.1} s", t0.elapsed().as_secs_f64());
}
